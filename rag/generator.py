"""Prompt assembly + thin call to the chat backend (Phi-3.5 Mini via Foundry Local, with
an Ollama fallback -- same abstraction pattern as embedder.py).

Citation format: the delivered UI design (static/index.html, adapted from the Claude
Design handoff bundle) renders answers as a list of `segments` -- alternating
`{"txt": "..."}` plain-text pieces and `{"c": n}` inline citation-chip references -- plus
a deduped `sources` list of `{"name", "url"}` that the chips and the footer pill-row both
point into. This is a step up from PROJECT_PLAN.md's original footer-only example (a
single "**Kaynaklar:**" list at the end): it's the same idea, closer to how Claude's own
citations render, and the design is the more concrete/current expression of what's wanted.

The context block shown to the model numbers *sources*, not retrieved chunks (a chunk may
carry more than one source -- see rag/chunker.py -- and if the model doesn't cite it, that
distinct source shouldn't take up a number). The model is instructed to inline-cite using
those same numbers as `[n]` right after the sentence that uses that source; `parse_segments`
then splits its raw text on that pattern into the `segments` shape the frontend expects.
Numbers the model invents that don't correspond to a real source are left as literal text
rather than crashing or pointing at nothing.

The Foundry Local chat-completion call below is verified against the real
foundry-local-sdk==0.5.1 source (see rag/embedder.py's module docstring and
requirements.txt for details on why that exact pre-1.0 version is pinned).
"""
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from . import config

SYSTEM_PROMPT_TR = """Sen, Firat Universitesi'nden Sam Houston State University'ye (SHSU) \
transfer olan ogrencilere yardimci olan bir asistansin.

Ogrenciye hitap ederken, adini bilmiyorsan (soru veya sohbet gecmisinde acikca soylenmedikce) \
ismiyle degil "Bearkat" diye hitap et -- SHSU'nun maskotu/lakabi, sicak ve samimi durur. \
Ogrenci kendi adini soylerse o andan itibaren adiyla hitap et.

Sadece asagida sana verilen kaynak metinlerdeki (Context) bilgilere dayanarak, Turkce ve \
net bir sekilde cevap ver. Kaynaklarda olmayan bilgiyi uydurma.

KISA cevap ver: en fazla 2-3 cumle. Gereksiz tekrar veya uzun aciklama ekleme, dogrudan \
soruyu yanitla.

Her kaynak metin basinda [1], [2] gibi bir numarayla etiketlenmis. Cevabinda bir kaynaktaki \
bilgiyi kullandiginda, ilgili cumlenin hemen sonuna o kaynagin numarasini koseli parantez \
icinde ekle (ornek: "Yillik maliyet yaklasik $41,860 [1]."). Ayni cumlede birden fazla \
kaynak kullandiysan hepsini ardarda ekle (ornek: "...gerekir [1][2]."). Numarayi yalnizca \
gercekten o bilgiyi verdiyse kullan.

Ogrenciyi bir eyleme yonlendiriyorsan (randevu almasi, bir siteye gitmesi, bir formu \
doldurmasi, bir ofise basvurmasi gerekiyorsa), o eylemin kaynagini MUTLAKA [n] ile isaretle \
-- ogrenci boylece dogrudan ilgili linke tiklayabilir. Yonlendirme yaparken sadece "ilgili \
ofise basvur" gibi belirsiz birakma; hangi kaynaktan oldugunu [n] ile belirt.

Eger soru bir bilgi talebi degil de "naber", "merhaba", "selam" gibi bir selamlasma/sohbet \
acilisiysa, kaynaklari yok sayip sicak ve kisa bir selamla karsilik ver, ardindan transfer \
sureciyle ilgili nasil yardimci olabilecegini sor. Bu durumda "bilgim yok" deme ve hicbir \
numara ekleme -- ortada cevaplanacak bir soru yok, sadece bir selamlasma var.

Eger soru gercek bir bilgi talebiyse ama cevabi verilen kaynaklarda yoksa, acikca "Bu \
konuda elimde bilgi yok." de ve hicbir numara ekleme.

Cevabinin sonuna ayrica bir kaynak listesi yazma -- kaynaklar, kullandigin [n] isaretlerine \
gore otomatik olarak gosterilecek. Sadece soruyu, gerektiginde inline [n] isaretleriyle \
cevapla."""

SYSTEM_PROMPT_EN = """You are an assistant helping students who are transferring from \
Firat University to Sam Houston State University (SHSU).

When addressing the student, if you don't know their name (unless they've stated it in the \
question or conversation), address them as "Bearkat" rather than by name -- SHSU's mascot/\
nickname, reads warm and friendly. If they tell you their name, use it from then on.

Answer clearly and only using the information in the source excerpts provided to you \
below (Context), in English. Do not invent information that isn't in the sources.

Keep it SHORT: 2-3 sentences at most. Don't repeat yourself or over-explain -- answer the \
question directly.

Each source excerpt is labeled with a number like [1], [2]. When you use information from \
a source, add that source's number in square brackets right after the sentence that uses \
it (example: "The annual cost is about $41,860 [1]."). If a sentence draws on more than \
one source, add all of them back to back (example: "...is required [1][2]."). Only use a \
number when that source actually supports the sentence.

If you're directing the student to take an action (book an appointment, visit a site, fill \
out a form, contact an office), ALWAYS cite that action's source with [n] so they get a \
clickable link straight to it. Don't leave a directive vague like "contact the relevant \
office" without a [n] pointing to which one.

If the message is a greeting or small talk ("hey", "hi", "what's up") rather than an actual \
question, ignore the sources and reply with a short, warm greeting, then ask how you can \
help with their transfer process. Don't say "I don't have information on that" for a \
greeting and don't add any citation numbers -- there's no question to answer yet.

If the message is a real question but its answer is not present in the given sources, \
clearly say "I don't have information on that." and don't add any citation numbers.

Do not append your own source list at the end of your answer -- the sources will be shown \
automatically based on the [n] markers you use. Just answer the question, citing inline \
with [n] where relevant."""


def get_system_prompt(language: str) -> str:
    """language is "tr" or "en" (anything else falls back to English)."""
    return SYSTEM_PROMPT_TR if language == "tr" else SYSTEM_PROMPT_EN


def _chunk_sources(chunk: Dict) -> List[Dict]:
    """A chunk's sources list, falling back to its source_file if chunker.py didn't
    attach one (shouldn't normally happen, but keeps this module standalone-safe)."""
    return chunk.get("sources") or [{"name": chunk.get("source_file") or "unknown source", "url": None}]


def _unique_sources(chunks: List[Dict]) -> List[Dict]:
    """Dedupe (name, url) pairs across every source in every retrieved chunk, preserving
    first-seen order. A single chunk may itself list more than one source."""
    seen = set()
    sources = []
    for chunk in chunks:
        for src in _chunk_sources(chunk):
            name = src.get("name") or chunk.get("source_file") or "unknown source"
            url = src.get("url")
            key = (name, url)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"name": name, "url": url})
    return sources


def build_context_and_sources(chunks: List[Dict]) -> Tuple[str, List[Dict]]:
    """Render retrieved chunks into a context block for the prompt, numbering by distinct
    *source* (not by chunk) so the numbers the model is asked to cite with `[n]` map 1:1
    onto the deduped `sources` list the frontend renders as citation chips/pills."""
    sources = _unique_sources(chunks)
    index = {(s["name"], s["url"]): i for i, s in enumerate(sources, start=1)}

    blocks = []
    for chunk in chunks:
        numbers = sorted(
            {
                index[(src.get("name") or chunk.get("source_file") or "unknown source", src.get("url"))]
                for src in _chunk_sources(chunk)
            }
        )
        label = "".join(f"[{n}]" for n in numbers)
        blocks.append(f"{label}\n{chunk.get('content') or chunk.get('text', '')}")
    return "\n\n".join(blocks), sources


def build_user_prompt(question: str, chunks: List[Dict]) -> str:
    context, _ = build_context_and_sources(chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


_CITATION_RE = re.compile(r"\[(\d+)\]")


def parse_segments(raw_answer: str, num_sources: int) -> List[Dict]:
    """Split the model's raw text into the frontend's alternating segment shape:
    `{"txt": "..."}` for plain text, `{"c": n}` for an inline citation chip referencing
    sources[n-1]. A `[n]` outside 1..num_sources isn't a valid citation (the model
    referenced a source number that doesn't exist) -- left as literal text rather than
    producing a chip that points nowhere."""
    segments: List[Dict] = []
    pos = 0
    for m in _CITATION_RE.finditer(raw_answer):
        n = int(m.group(1))
        if not (1 <= n <= num_sources):
            continue
        if m.start() > pos:
            segments.append({"txt": raw_answer[pos : m.start()]})
        segments.append({"c": n})
        pos = m.end()
    if pos < len(raw_answer):
        segments.append({"txt": raw_answer[pos:]})
    return segments


class ContentBlocked(Exception):
    """Raised by a ChatBackend.generate() when the backend's own safety system
    refused to produce a response -- distinct from a backend outage/bug, so callers
    can show a deliberate "I won't do that" message instead of a generic error."""


class ChatBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        """Return the model's raw answer text for the given system+user prompt.
        `max_tokens` bounds output length -- a real speed lever for local inference
        (generation time scales with output token count), not just a quality knob."""
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

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        self._ensure_client()
        # Qwen3 is a hybrid-thinking model that defaults to emitting raw chain-of-thought
        # text instead of a final answer (same failure mode as GeminiChat's thinking bug --
        # with short max_tokens caps, the answer gets cut off mid-reasoning). Qwen3's chat
        # template recognizes a literal "/no_think" directive in the prompt to switch it
        # into direct-answer mode. Other models (phi-3.5-mini, etc.) just ignore the
        # literal text, so this is safe to send unconditionally to any qwen3 alias.
        if self.model_name.lower().startswith("qwen3"):
            system_prompt = f"{system_prompt} /no_think"
        kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


class OllamaChat(ChatBackend):
    """Fallback chat backend using Ollama's local HTTP API (PROJECT_PLAN.md "Yedek Plan")."""

    def __init__(self, model_name: str = None, host: str = None):
        self.model_name = model_name or config.OLLAMA_CHAT_MODEL
        self.host = host or config.OLLAMA_HOST

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        import requests

        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if max_tokens is not None:
            body["options"] = {"num_predict": max_tokens}

        resp = requests.post(f"{self.host}/api/chat", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class GeminiChat(ChatBackend):
    """Gemini 2.5 Flash via the Gemini API -- optional cloud-deploy path (RAG_CHAT_BACKEND=gemini),
    not the offline default. `genai.Client()` reads GEMINI_API_KEY from the environment itself."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.GEMINI_CHAT_MODEL
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return

        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Run `pip install google-genai` "
                "(see requirements.txt), or set RAG_CHAT_BACKEND=foundry/ollama instead."
            ) from e

        self._client = genai.Client()

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        self._ensure_client()
        from google.genai import types

        # Gemini 2.5 Flash thinks by default, and thinking tokens count against
        # max_output_tokens -- with the short caps this project uses (see config.py's
        # MAX_ANSWER_TOKENS/MAX_TRANSLATE_TOKENS), that silently starves the visible
        # answer (finish_reason=MAX_TOKENS, response.text=None) before it's written.
        # This is a short-citation-answer bot, not a reasoning task, so thinking is
        # disabled outright rather than budgeted around.
        config_kwargs = {
            "system_instruction": system_prompt,
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        # response.text is None whenever Gemini's own safety layer declined to answer
        # (finish_reason SAFETY/PROHIBITED_CONTENT/etc, empty content.parts) -- with
        # thinking disabled above, a normal answer always has text, so None here means
        # "refused," not "empty." Surface that distinctly instead of returning None
        # (which would blow up downstream in parse_segments expecting a string).
        if response.text is None:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            raise ContentBlocked(f"Gemini declined to respond (finish_reason={finish_reason})")
        return response.text


def get_chat_backend(backend: str = None) -> ChatBackend:
    """Factory: returns the configured ChatBackend implementation.

    `backend` defaults to config.CHAT_BACKEND ("foundry", "ollama", or "gemini"), overridable
    via the RAG_CHAT_BACKEND env var or by passing it explicitly.
    """
    backend = (backend or config.CHAT_BACKEND).lower()
    if backend == "foundry":
        return FoundryLocalChat()
    if backend == "ollama":
        return OllamaChat()
    if backend == "gemini":
        return GeminiChat()
    raise ValueError(f"Unknown chat backend: {backend!r} (expected 'foundry', 'ollama', or 'gemini')")


_TRANSLATE_SYSTEM_PROMPT = (
    "Translate the user's message to English. Reply with ONLY the translated text -- "
    "no quotes, no explanation, no extra commentary. If it's already in English, repeat "
    "it unchanged."
)


def translate_query_for_retrieval(question: str, backend: ChatBackend) -> str:
    """Translate a non-English question to English before embedding it for retrieval.

    Why this exists: the knowledge base (docs/) is English-only, and measured directly
    against this project's real docs/knowledge.db, the local embedding models actually
    available (nomic-embed-text, mxbai-embed-large via Ollama) retrieve poorly for a
    Turkish query -- e.g. "yillik maliyet ne kadar?" scored ~0.40 similarity against every
    candidate chunk (no clear winner, effectively noise), while the same question translated
    to English scored 0.53-0.65 against the correct chunk. Translating first decouples
    retrieval quality from a given embedding backend's cross-lingual ability -- useful for
    the Ollama fallback today, and cheap insurance if qwen3-embedding-0.6b via Foundry Local
    turns out to need it too. The ORIGINAL question (not this translation) is still what
    gets passed to answer_query, so the model answers in the user's own language/phrasing.
    """
    translated = backend.generate(
        _TRANSLATE_SYSTEM_PROMPT, question, max_tokens=config.MAX_TRANSLATE_TOKENS
    ).strip()
    return translated or question


def answer_query(
    question: str,
    chunks: List[Dict],
    language: str = "tr",
    backend: Optional[ChatBackend] = None,
) -> Dict:
    """Assemble the prompt from retrieved chunks + question, call the chat backend, and
    return {"segments": [...], "sources": [...]} -- the shape static/index.html's chat UI
    renders directly (inline [n] citation chips + a deduped source pill list)."""
    backend = backend or get_chat_backend()
    system_prompt = get_system_prompt(language)
    context, sources = build_context_and_sources(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    raw_answer = backend.generate(system_prompt, user_prompt, max_tokens=config.MAX_ANSWER_TOKENS)
    segments = parse_segments(raw_answer, num_sources=len(sources))

    # `sources` above is every retrieved chunk's source, not just the ones the model
    # actually cited (e.g. it retrieved chunks but answered "I don't have information on
    # that," or this is a greeting with no citations at all) -- showing all of them next to
    # an uncited answer reads as a bug (a pile of "sources" for a non-answer). Keep only
    # the cited ones, renumbered so the chip numbers stay sequential and gap-free.
    cited_numbers = sorted({seg["c"] for seg in segments if "c" in seg})
    cited_sources = [sources[n - 1] for n in cited_numbers]
    renumber = {old_n: new_n for new_n, old_n in enumerate(cited_numbers, start=1)}
    segments = [{"c": renumber[seg["c"]]} if "c" in seg else seg for seg in segments]

    return {"segments": segments, "sources": cited_sources}
