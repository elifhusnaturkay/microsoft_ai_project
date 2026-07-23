"""Prompt assembly + thin call to the chat backend (Phi-3.5 Mini via Foundry Local, with
an Ollama fallback -- same abstraction pattern as embedder.py).

Citation format: PROJECT_PLAN.md's locked example answers show a footer-style citation
block, e.g. (Turkish):

    ...answer text...

    **Kaynaklar:**
    - [SHSU Cost of Attendance](https://www.shsu.edu/cost-aid/cost-attendance)
    - [Cashiering Services - Payments](https://www.shsu.edu/offices-departments/...)

and (English):

    ...answer text...

    **Sources:**
    - [SHSU Cost of Attendance](https://www.shsu.edu/cost-aid/cost-attendance)

Rather than asking the LLM to reproduce document names/URLs from memory (which risks
hallucinated links), that footer is assembled programmatically in `answer_query` from the
*actual* retrieved chunks' source_name/source_url metadata, and appended to whatever the
model generates. The system prompt asks the model to focus on answering from context and
not invent a sources list of its own.

The Foundry Local chat-completion call below is verified against the real
foundry-local-sdk==0.5.1 source (see rag/embedder.py's module docstring and
requirements.txt for details on why that exact pre-1.0 version is pinned).
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from . import config

SYSTEM_PROMPT_TR = """Sen, Firat Universitesi'nden Sam Houston State University'ye (SHSU) \
transfer olan ogrencilere yardimci olan bir asistansin.

Sadece asagida sana verilen kaynak metinlerdeki (Context) bilgilere dayanarak, Turkce ve \
net bir sekilde cevap ver. Kaynaklarda olmayan bilgiyi uydurma.

Eger sorunun cevabi verilen kaynaklarda yoksa, acikca "Bu konuda elimde bilgi yok." de.

Cevabinin sonuna kendi kaynak listeni ekleme -- kaynaklar ayrica, kullandigin kaynak \
metinlere dayanarak otomatik olarak eklenecek. Sadece soruyu cevapla."""

SYSTEM_PROMPT_EN = """You are an assistant helping students who are transferring from \
Firat University to Sam Houston State University (SHSU).

Answer clearly and only using the information in the source excerpts provided to you \
below (Context), in English. Do not invent information that isn't in the sources.

If the answer to the question is not present in the given sources, clearly say \
"I don't have information on that."

Do not append your own source list at the end of your answer -- the sources will be \
added automatically based on the context you were given. Just answer the question."""


def get_system_prompt(language: str) -> str:
    """language is "tr" or "en" (anything else falls back to English)."""
    return SYSTEM_PROMPT_TR if language == "tr" else SYSTEM_PROMPT_EN


def _format_source_label(source: Dict) -> str:
    name = source.get("name") or "unknown source"
    return f"{name} ({source['url']})" if source.get("url") else name


def format_context_with_sources(chunks: List[Dict]) -> str:
    """Render retrieved chunks into a numbered context block for the prompt. A chunk may
    carry more than one source (chunker.py merges small adjacent sections together) --
    all of them are listed so the model sees exactly what it's allowed to cite."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        sources = chunk.get("sources") or []
        if sources:
            label = "; ".join(_format_source_label(src) for src in sources)
        else:
            label = chunk.get("source_file") or "unknown source"
        blocks.append(f"[{i}] {label}\n{chunk.get('content') or chunk.get('text', '')}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: List[Dict]) -> str:
    context = format_context_with_sources(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def _unique_sources(chunks: List[Dict]) -> List[Dict]:
    """Dedupe (name, url) pairs across every source in every retrieved chunk, preserving
    first-seen order. A single chunk may itself list more than one source."""
    seen = set()
    sources = []
    for chunk in chunks:
        chunk_sources = chunk.get("sources") or [
            {"name": chunk.get("source_file") or "unknown source", "url": None}
        ]
        for src in chunk_sources:
            name = src.get("name") or chunk.get("source_file") or "unknown source"
            url = src.get("url")
            key = (name, url)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"name": name, "url": url})
    return sources


def format_citation_footer(chunks: List[Dict], language: str) -> str:
    """Build the '**Kaynaklar:**' / '**Sources:**' markdown footer from real chunk metadata,
    replicating the exact format of PROJECT_PLAN.md's example answers."""
    sources = _unique_sources(chunks)
    if not sources:
        return ""

    heading = "**Kaynaklar:**" if language == "tr" else "**Sources:**"
    lines = [heading]
    for src in sources:
        if src["url"]:
            lines.append(f"- [{src['name']}]({src['url']})")
        else:
            lines.append(f"- {src['name']}")
    return "\n".join(lines)


class ChatBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw answer text for the given system+user prompt."""
        raise NotImplementedError


class FoundryLocalChat(ChatBackend):
    """Phi-3.5 Mini via Microsoft Foundry Local. See module TODO above."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.LOCAL_CHAT_MODEL
        self._client = None
        self._model_id = None

    def _ensure_client(self):
        if self._client is not None:
            return

        try:
            from foundry_local import FoundryLocalManager
        except ImportError as e:
            raise RuntimeError(
                "foundry-local-sdk is not installed. Run `pip install foundry-local-sdk` "
                "(see requirements.txt), or set RAG_CHAT_BACKEND=ollama to use the fallback."
            ) from e
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The `openai` package is required to talk to Foundry Local's local "
                "OpenAI-compatible endpoint. Run `pip install openai`."
            ) from e

        manager = FoundryLocalManager(self.model_name)
        self._model_id = manager.get_model_info(self.model_name, raise_on_not_found=True).id
        self._client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class OllamaChat(ChatBackend):
    """Fallback chat backend using Ollama's local HTTP API (PROJECT_PLAN.md "Yedek Plan")."""

    def __init__(self, model_name: str = None, host: str = None):
        self.model_name = model_name or config.OLLAMA_CHAT_MODEL
        self.host = host or config.OLLAMA_HOST

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import requests

        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def get_chat_backend(backend: str = None) -> ChatBackend:
    """Factory: returns the configured ChatBackend implementation.

    `backend` defaults to config.CHAT_BACKEND ("foundry" or "ollama"), overridable via the
    RAG_CHAT_BACKEND env var or by passing it explicitly.
    """
    backend = (backend or config.CHAT_BACKEND).lower()
    if backend == "foundry":
        return FoundryLocalChat()
    if backend == "ollama":
        return OllamaChat()
    raise ValueError(f"Unknown chat backend: {backend!r} (expected 'foundry' or 'ollama')")


def answer_query(
    question: str,
    chunks: List[Dict],
    language: str = "tr",
    backend: Optional[ChatBackend] = None,
) -> Dict:
    """Assemble the prompt from retrieved chunks + question, call the chat backend, and
    return {"answer": <markdown text with citation footer>, "sources": [{"name", "url"}]}.
    """
    backend = backend or get_chat_backend()
    system_prompt = get_system_prompt(language)
    user_prompt = build_user_prompt(question, chunks)

    model_answer = backend.generate(system_prompt, user_prompt)
    footer = format_citation_footer(chunks, language)
    full_answer = f"{model_answer}\n\n{footer}" if footer else model_answer

    return {"answer": full_answer, "sources": _unique_sources(chunks)}
