# SHSU Transfer Student AI Assistant — Proje Plani

> Microsoft Foundry Local ile tamamen offline calisan, Turkce/Ingilizce bilingual Q&A chatbot

---

## Kesinlesmis Kararlar

| Karar | Secim | Neden |
|-------|-------|-------|
| **Dokuman icerigi** | Sam Houston State University belgeleri | Gercek universite verisi, ogrenciler icin anlamli |
| **Arayuz** | Streamlit (web UI) | Gorsel, demo icin ideal, ogrencilere gosterilir |
| **Dil** | Turkce + Ingilizce (kullanici secer) | Turk transfer ogrencileri icin Turkce, genel kullanim icin Ingilizce |
| **Platform** | macOS + Windows | Her iki platformda da calisacak |
| **Kaynak gosterme** | Claude tarzi inline referans | Cevap icinde dokuman adi + tiklanabilir URL |
| **Chat modeli** | Phi-3.5 Mini | Hiz + 16GB RAM'e uygun |
| **Embedding modeli** | qwen3-embedding-0.6b | Lokal, offline, hafif |
| **Chunk stratejisi** | Paragraf bazli (~200-500 kelime) | Dogal metin sinirlari |
| **Top-K** | 3 chunk | Yeterli context, fazla gurultu yok |
| **Yedek plan** | Ollama | Foundry Local sorun cikarirsa gecis yapilir |
| **Web deploy** | Streamlit Community Cloud | Bedava, `appname.streamlit.app` URL, domain gerektirmez |
| **Web LLM** | Google Gemini Flash (free tier) | Turkce destegi mukemmel, 15 req/dk bedava |
| **Kod yapisi** | Tek codebase, config ile local/cloud secimi | Ayni RAG pipeline, sadece LLM katmani degisir |

---

## Proje Nedir?

Sam Houston State University belgelerinden sorulara cevap verebilen, **internet baglantisi gerektirmeyen**, **Turkce ve Ingilizce** destekleyen bir chatbot. Tum AI islemi kullanicinin kendi bilgisayarinda (macOS veya Windows) gerceklesir.

### Dil Destegi

Kullanici Streamlit arayuzunde dil secebilir (TR/EN). Secime gore:
- **System prompt** dili degisir (Turkce veya Ingilizce yanit uretmesi icin)
- **UI etiketleri** degisir (butonlar, basliklar, placeholder'lar)
- Bilgi tabani (docs/) her iki dilde de aynidir — model secilen dilde cevap uretir

### Kaynak Gosterme (Citation)

Her chunk'a kaynak bilgisi eklenir: `source_name` + `source_url`. Model cevap verirken bunlari Claude tarzi inline referans olarak gosterir.

**Turkce cevap ornegi:**
> Uluslararasi ogrenciler icin yillik tahmini maliyet yaklasik $41,860'dir. Bu tutar okul ucreti, barinma, yemek ve kisisel giderleri kapsar. Odemeyi online olarak kredi karti, TransferMate veya Convera ile yapabilirsiniz.
>
> **Kaynaklar:**
> - [SHSU Cost of Attendance](https://www.shsu.edu/cost-aid/cost-attendance)
> - [Cashiering Services - Payments](https://www.shsu.edu/offices-departments/student-account-services/cashiering-services/payments)

**English answer example:**
> The estimated annual cost for international students is approximately $41,860. This includes tuition, housing, meals, and personal expenses. You can pay online via credit card, TransferMate, or Convera.
>
> **Sources:**
> - [SHSU Cost of Attendance](https://www.shsu.edu/cost-aid/cost-attendance)
> - [Cashiering Services - Payments](https://www.shsu.edu/offices-departments/student-account-services/cashiering-services/payments)

### Calisma Prensibi (RAG Pipeline)

```
Kullanici Sorusu
  ↓
[1] Soru → embedding vektorune cevrilir (qwen3-embedding-0.6b)
  ↓
[2] SQLite'ta cosine similarity ile en benzer chunk'lar bulunur
  ↓
[3] Soru + bulunan chunk'lar birlestirilerek prompt olusturulur
  ↓
[4] Foundry Local LLM'e gonderilir (Phi-3.5 Mini)
  ↓
[5] Model cevabi kullaniciya dondurulur (kaynak gostererek)
```

### Teknik Mimari

```
┌─────────────────────────────────────────────────┐
│                 Kullanici Arayuzu               │
│               (Streamlit Web UI)                 │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              RAG Pipeline (Python)               │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Ingestion│  │ Retrieval │  │  Generation  │  │
│  │ (chunk + │  │ (query →  │  │ (context +   │  │
│  │ embed +  │  │ top-K     │  │  soru → LLM  │  │
│  │ store)   │  │ chunks)   │  │  → cevap)    │  │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  │
│       │              │               │           │
└───────┼──────────────┼───────────────┼───────────┘
        │              │               │
┌───────▼──────┐       │        ┌──────▼───────┐
│   SQLite DB  │◄──────┘        │ Foundry Local│
│  (chunks +   │                │  LLM Runtime │
│  embeddings) │                │ (Phi-3.5 Mini)│
└──────────────┘                └──────────────┘
```

---

## Gereksinimler

### Donanim
- **macOS** (Apple Silicon veya Intel) veya **Windows** (10/11)
- Minimum 8GB RAM (16GB tavsiye)
- ~5GB bos disk alani (model indirmeleri icin)

### Yazilim (Kurulmasi Gerekenler)

| Arac | Neden | macOS Kurulum | Windows Kurulum |
|------|-------|---------------|-----------------|
| **Python 3.10+** | Ana gelistirme dili | `brew install python@3.12` | python.org'dan indir |
| **Foundry Local SDK** | LLM ve embedding | `pip install foundry-local-sdk` | Ayni |
| **SQLite3** | Vektor veritabani | Python ile birlikte gelir | Ayni |
| **Streamlit** | Web arayuzu | `pip install streamlit` | Ayni |

### Python Kutuphaneleri (requirements.txt)

```
foundry-local-sdk
numpy
streamlit
```

### Bilgi Tabani (docs/ Klasoru)

Chatbot'un cevaplayacagi konular ve kaynaklari. Hedef kitle: **Firat Universitesi'nden SHSU'ya transfer olan Software Engineering ogrencileri**.

Referans site: https://umawi1427.github.io/shsu-transfer-process/ (26 adimlik transfer rehberi)

#### Dokuman 1: `transfer_process.txt` — Transfer Sureci (Adim Adim)
Kaynak: Arkadasin sitesi (step0-step25) + SHSU resmi sayfalari
- Duolingo English Test (min 100 puan)
- ApplyTexas hesabi olusturma (International Transfer secimi)
- SHSU Self-Service Center kaydi
- Dokuman yukleme (transkript, Duolingo skoru, $90 basvuru ucreti)
- SHSU email aktivasyonu + Duo kimlik dogrulama
- I-20 basvurusu (pasaport + min $39,110 banka hesap ozeti)
- TB anketi ve menenjit asisi
- SEVIS ucreti ($350)
- DS-160 formu + vize randevusu
- Vize mulakat dokumanlari ve sik sorulan sorular

#### Dokuman 2: `tuition_fees.txt` — Ucretler ve Odeme
Kaynak: shsu.edu/cost-aid/ + catalog.shsu.edu
- Uluslararasi ogrenci ucretleri ($455/kredi saat tuition + $194 designated + $80 institutional)
- Yillik tahmini maliyet (~$41,860)
- Odeme yontemleri (online, TransferMate, Convera, kredi karti)
- Taksit plani ($30 kayit ucreti)
- Burs basvurusu (Scholarships4Kats + Firat ozel formu)

#### Dokuman 3: `course_registration.txt` — Ders Kaydi ve Danisman
Kaynak: catalog.shsu.edu + samweb.shsu.edu
- Danisman randevusu alma (SAM Center)
- SamWeb uzerinden ders kaydi
- TSI sinavi (Texas Success Initiative) gerekliligi ve muafiyet
- Akademik takvim ve donemler
- Title IX egitimi zorunlulugu
- Bearkat Bundle opt-out

#### Dokuman 4: `credit_transfer.txt` — Kredi Transferi ve Ders Saydirma
Kaynak: TESS + SACSCOC 10.8 + catalog
- Firat-SHSU 2+2 programi (2010'dan beri)
- Transfer Equivalency Self-Service (TESS) araci
- Maksimum 70 transfer kredi saati
- Texas Core Curriculum (42 saat) gerekliligi
- Fakulte degerlemesi sureci (syllabus karsilastirma)
- Ders crosswalk tablolari (CS ve Matematik)

#### Dokuman 5: `software_engineering.txt` — Software Engineering Programi
Kaynak: catalog.shsu.edu + cs.shsu.edu
- BS Software Engineering — 120 kredi saat
- Core Curriculum (42 saat) + Degree Requirements (17 saat math)
- Major Foundation (45 saat, 16 COSC dersi)
- Prescribed Electives (6 saat) + General Electives (5 saat)
- Min 2.0 GPA (genel ve major)
- Computer Science bolumu ve fakultesi

#### Dokuman 6: `visa_immigration.txt` — Vize ve Gocmenlik
Kaynak: ISSS + arkadasin sitesi
- F-1 vize sureci (DS-160, mulakat, gerekli belgeler)
- I-20 nasil alinir
- SEVIS kaydi ve transfer
- CPT/OPT calisma izinleri (12 ay + STEM uzatma)
- On-campus calisma kurallari (20 saat/hafta)
- Saglik sigortasi zorunlulugu
- SSN basvurusu (Conroe, TX ofisi)

#### Dokuman 7: `bearkat_life.txt` — Kampus Yasami
Kaynak: shsu.edu/campus-life/ + arkadasin sitesi
- Yurt secenekleri (16 yurt, on/off-campus) + ucretler
- Yemek planlari (All Access, 15/hafta, 10/hafta, commuter)
- Bearkat OneCard (ogrenci kimlik + yemek + erisim)
- 250+ ogrenci kulubu (BearkatHQ)
- Spor ve rekreasyon (tirrmanma duvari, havuz, fitness)
- Kat Tracks shuttle servisi (HEB, Walmart, Target)
- Huntsville sehri (IAH havalimani'na <1 saat)
- Banka hesabi acma (Bank of America + TEB anlasma)
- Texas ehliyeti basvurusu

#### Dokuman 8: `orientation_arrival.txt` — Oryantasyon ve Varis
Kaynak: GEC + arkadasin sitesi
- Ucus rezervasyonu (IAH havalimani, I-20'den 30 gun once giris)
- Havalimani karsılama servisi
- Uluslararasi ogrenci oryantasyonu (zorunlu, 2 kisim)
  - Online: Blackboard 12 modul (%80+ quiz puani)
  - Yuzyuze: Farrington Building (pasaport + I-20 + I-94 getir)
- Honors College basvurusu (3.25+ GPA)
- Career Success Center

#### Dokuman 9: `financial_guide.txt` — Maddiyat Rehberi (Gelmeden Once + Yasam Masraflari)
Kaynak: shsu.edu + arastirma verileri
- **Gelmeden once gereken para:**
  - I-20 icin banka bakiyesi: ~$41,860 (1 yillik attendance cost)
  - Basvuru ucretleri toplam: ~$625 (ApplyTexas $90 + SEVIS $350 + vize $185)
  - Ucak bileti (Istanbul-Houston): $430-$750
  - Varis oncesi toplam harcama: $2,930-$3,700
  - Varista elde olmasi gereken nakit: $2,500-$4,000
- **Aylik yasam masraflari Huntsville'de:**
  - On-campus yurt: $2,520-$4,772/donem
  - Yemek plani: $1,960-$2,750/donem
  - Off-campus paylasimli daire: $439-$699/ay (The Grove $439-$569, Encore $449-$699, Villas $559-$579)
  - Market/bakkaliye: $300-$400/ay
  - Telefon hatti (T-Mobile prepaid): $45-$60/ay
  - Saglik sigortasi: ~$3,250/yil (~$1,625/donem, zorunlu)
  - Arac sigortasi (varsa): $43-$195/ay
- **Ornek aylik butce:**
  - On-campus senaryo: ~$1,860/ay
  - Off-campus senaryo: ~$1,355/ay

#### Dokuman 10: `oncampus_jobs.txt` — Kampuste Is Bulma Rehberi
Kaynak: shsu.edu + PeopleAdmin + GEC
- **Nasil is bulunur:**
  - shsu.peopleadmin.com → "Student" pozisyonlari filtrele
  - Aramark (yemek servisi) — aramark.com'da "Sam Houston" ara
  - Bolum asistanliklari, kutuphane, IT help desk, tutorluk
- **Maas araligi:** $10-$16.57/saat
  - 20 saat/hafta x $12/saat = ~$960/ay
  - Yaz donemi: 28 saat/haftaya kadar (3 kredi kayitliysa)
- **Ise baslamak icin gereken surec:**
  - Varis → 10 gun bekle → is teklifi al → SSN basvurusu (Conroe SSA)
  - SSN karti: 2-4 haftada gelir
  - Ilk maas: varistan ~4-6 hafta sonra
- **Pratik is bulma ipuclari:**
  - Donem baslamadan once basvur (Agustos basi)
  - Dining/Aramark en cok alan yer
  - Resume hazirla (Career Success Center yardim eder)
  - Bolum hocalarina lab/research assistant sor

#### Dokuman 11: `scholarships_financial_aid.txt` — Burslar ve Mali Destek
Kaynak: shsu.edu + COSET + Firat
- **Otomatik burslar:**
  - Bearkat Transfer Scholarship: $2,000/yil (2.75+ GPA, 30-90 kredi)
- **Basvuru gerektiren burslar:**
  - Scholarships4Kats portali (shsu.academicworks.com)
  - COSET Recruitment Scholarship: $2,000 (ilk yil)
  - Honors Transfer Scholarship: $5,000 (tek seferlik, 3.25+ GPA)
- **Firat Universitesi ozel bursu:**
  - Haziran basinda form doldur
  - mkarabatak@firat.edu.tr ve pdmartin@shsu.edu'ya gonder
  - Ardindan Scholarships4Kats'tan genel basvuruyu tamamla
- **Non-Resident Tuition Waiver:**
  - $1,000+ rekabetci SHSU bursu alirsan → in-state ucretine duser
  - Potansiyel tasarruf: $6,000-$8,000/yil
- **Acil durum fonu:**
  - Bearkat Emergency Fund — mali sikintidaki tum kayitli ogrenciler icin

#### Dokuman 12: `offcampus_housing.txt` — Off-Campus Konut Rehberi (2026)
Kaynak: SHSU Residence Life — Huntsville Area Off Campus Housing List 2026
PDF: docs/shsu_offcampus_housing_2026.pdf

**~40 apartman/konut secenegi, her biri icin:**
- Isim, adres, telefon, website
- Yatak odasi sayisi, kira araligi, depozito
- Dahil olan hizmetler (elektrik, su, internet, kablo TV)
- Olanaklar (mobilyali mi, havuz, camasir, otopark, pet policy)
- Kampuse uzaklik ve shuttle durumu

**En ucuz secenekler (ogrenci butcesine uygun):**
- The Grove: $439-$569/ay, 2-3 BR, 1.8 mi, $10 valet parking
- Encore Sam Houston: $449-$699/ay, 4 BR, 1.1 mi
- Forest Gate: $629-$1,279/ay, 1-3 BR, 2.6 mi
- The Villas on Sycamore: $559-$579/ay, 4 BR, mobilyali, 0.4 mi
- Lark Huntsville: $429-$1,409/ay, 2-5 BR, 0.5 mi
- Magnolia West: $600-$724/ay, 1-3 BR, on-site laundry, 0.2 mi

**Kampuse en yakin (yuruyus mesafesi):**
- Bearkat Cottages: 0.1 mi, $540-$850
- The Arbors: 0.2 mi, $379-$1,318
- Magnolia West: 0.2 mi, $600-$724
- The Grove: 1.8 mi ama shuttle var
- Cornerstone: 0.1 mi, $849-$949
- Domain: 2.3 mi (MTM lease, esnek)

**Ek bilgiler:**
- Bireysel kiralama (individual lease) vs geleneksel kiralama farki
- Roommate matching hizmeti sunan yerler
- Mobilyali (furnished) secenekler
- Ek kiralik hizmetler: duplex, ev, mobil ev, RV alanlari (~30 ek saglayici)
- Fiyatlar degisebilir — SHSU onay vermez, sadece bilgi amacli paylasilir

#### Dokuman 13: `faq_practical.txt` — Sik Sorulan Pratik Sorular
Kaynak: Derleme (tum kaynaklar)
- Gelmeden once ne kadar para biriktirmeliyim?
- Aylik ne kadar harcama yaparim?
- Is bulamazsam ne olur?
- Hangi yurdu secmeliyim?
- Off-campus mi on-campus mi daha ucuz?
- Araba almali miyim?
- Yemek plani zorunlu mu?
- Saglik sigortasi ne kadar ve zorunlu mu?
- Burs kazanamazsam ne olur?
- Kredi kartim Turkiye'den calisir mi? (TEB-Bank of America anlasma)
- Ilk hafta nelere dikkat etmeliyim?
- Donem arasi (yaz) ne yaparim?
- Mezun olduktan sonra calisabilir miyim? (OPT 12 ay + STEM 24 ay)
- Ailem ziyarete gelebilir mi?
- Huntsville'de Turk yemegi/marketi var mi?

### Referans Kaynaklar (Kod Gelistirme)

| Kaynak | Aciklama |
|--------|----------|
| **Arkadasin sitesi** | https://umawi1427.github.io/shsu-transfer-process/ — 26 adimlik transfer rehberi |
| **SHSU resmi sitesi** | shsu.edu — tum resmi bilgiler |
| **Microsoft Tech Community blog** | [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) |
| **Microsoft Learn tutorial** | "Build a RAG application" — resmi ornek kod |

---

## Uygulama Adimlari

### Adim 1: Ortam Kurulumu

**macOS:**
```bash
cd ~/Development/microsoft_ai_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```powershell
cd C:\Development\microsoft_ai_project
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Foundry Local testi:
```python
python -c "
from foundry_local import FoundryLocalManager
manager = FoundryLocalManager()
model = manager.get_model('phi-3.5-mini')
print('Model yuklendi:', model)
"
```

**Basari kriteri:** "Model yuklendi" mesajini gormek.

### Adim 2: Ingestion Pipeline

Dosya: `ingest.py`

Yapilacaklar:
1. Dokumanlari oku (txt dosyalarindan)
2. Her dokumani chunk'lara bol (paragraf bazli, ~1-3 paragraf)
3. Her chunk icin embedding uret (`qwen3-embedding-0.6b`)
4. SQLite'a kaydet — **kaynak bilgisiyle birlikte**

SQLite tablosu:
```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_file TEXT,      -- "financial_guide.txt"
    source_name TEXT,      -- "SHSU Cost of Attendance"
    source_url TEXT,       -- "https://www.shsu.edu/cost-aid/cost-attendance"
    content TEXT,          -- chunk icerigi
    embedding BLOB         -- vektor (JSON-serialized)
);
```

**Basari kriteri:** `knowledge.db` dosyasi olusmus, icinde chunk + embedding + kaynak bilgisi var.

### Adim 3: Retrieval Fonksiyonu

Dosya: `retrieval.py`

```python
def get_top_chunks(query: str, k: int = 3) -> list[dict]:
    """Sorguya en yakin k chunk'i kaynak bilgisiyle dondurur."""
    # Donen her chunk: {content, source_file, source_name, source_url, similarity}
    ...
```

**Basari kriteri:** Bilinen bir soru soruldiginda dogru dokumandan chunk + kaynak donuyor.

### Adim 4: LLM Entegrasyonu

Dosya: `generate.py`

```python
def answer_query(question: str, language: str = "tr") -> dict:
    """RAG pipeline: retrieval + generation + kaynak listesi."""
    chunks = get_top_chunks(question)
    context = format_context_with_sources(chunks)
    system_prompt = get_system_prompt(language)  # TR veya EN
    # LLM cevap uretir + kaynak listesi dondurulur
    return {"answer": "...", "sources": [{"name": "...", "url": "..."}]}
```

System prompt icinde kaynak gosterme talimati:
```
Cevabini verdikten sonra, kullandigin kaynaklari listele.
Eger dokumanlarda bilgi yoksa, "Bu konuda elimde bilgi yok" de.
```

**Basari kriteri:** `answer_query("ucretler ne kadar?", "tr")` → Turkce cevap + kaynaklar.

### Adim 5: Streamlit Arayuzu (Bilingual)

Dosya: `app.py`

Ozellikler:
1. **Dil secici** — sidebar'da TR/EN toggle
2. Soru giris alani
3. Cevap gosterimi + inline kaynaklar (tiklanabilir linkler)
4. Chat gecmisi (session state)

```python
import streamlit as st

# Sidebar: dil secimi
lang = st.sidebar.radio("Language / Dil", ["Turkce", "English"])

st.title("SHSU AI Assistant" if lang == "English" else "SHSU AI Asistani")
question = st.chat_input("Ask a question..." if lang == "English" else "Sorunuzu yazin...")

if question:
    result = answer_query(question, "en" if lang == "English" else "tr")
    st.markdown(result["answer"])
    with st.expander("Sources / Kaynaklar"):
        for src in result["sources"]:
            st.markdown(f"- [{src['name']}]({src['url']})")
```

**Basari kriteri:** `streamlit run app.py` → tarayicida TR/EN chatbot, kaynakli cevaplar.

### Adim 6: Config — Local/Cloud Secimi

Dosya: `config.py`

```python
MODE = "local"  # "local" veya "cloud"

# Local mode (Foundry Local — yaz okulu isterlerini karsilar)
LOCAL_CHAT_MODEL = "phi-3.5-mini"
LOCAL_EMBED_MODEL = "qwen3-embedding-0.6b"

# Cloud mode (Web deploy — ayni RAG pipeline, farkli LLM)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # .env veya Streamlit secrets
GEMINI_MODEL = "gemini-2.0-flash"

DB_PATH = "knowledge.db"
TOP_K = 3
```

`generate.py` icinde:
```python
if config.MODE == "local":
    # Foundry Local ile cevap uret
elif config.MODE == "cloud":
    # Google Gemini API ile cevap uret
```

### Adim 7: Web Deploy (Streamlit Community Cloud)

1. GitHub repo olustur (public)
2. `requirements.txt` + `app.py` + `knowledge.db` push et
3. [share.streamlit.io](https://share.streamlit.io) → GitHub repo'yu bagla
4. Streamlit Secrets'a `GEMINI_API_KEY` ekle
5. Deploy → `shsu-assistant.streamlit.app` gibi bedava URL

```
# .streamlit/secrets.toml (Streamlit Cloud'da Secrets UI'dan eklenir)
GEMINI_API_KEY = "..."
```

**Basari kriteri:** LinkedIn'e paylasabilcegin calisan bir link.

### Adim 8: Cilalama

- [ ] Hata yonetimi (bos sorgu, model yuklenemezse, DB bos ise)
- [ ] "Bilmiyorum" senaryosu (dokumanlarda olmayan sorular)
- [ ] Performans kontrolu (lokal: 1-3sn, cloud: <2sn)
- [ ] Windows uyumluluk testi (path separator, encoding)

### Adim 9: Ogretim Materyali

- [ ] Her hafta icin ogrenci egzersiz dosyalari
- [ ] Adim adim kurulum rehberi (macOS + Windows)
- [ ] Demo senaryolari (sunumda gosterilecek sorular)
- [ ] README.md (projenin ne oldugu, nasil calistirilacagi)

---

## Proje Dosya Yapisi (Hedef)

```
microsoft_ai_project/
├── PROJECT_PLAN.md          ← Bu dosya
├── BELGE_TOPLAMA_REHBERI.md ← Kaynak URL haritasi
├── Summer School Foundry Local Plan.pdf
├── requirements.txt
├── .env                     ← API key'ler (git'e eklenmez)
├── .gitignore
├── main.py                  ← CLI giris noktasi (test icin)
├── ingest.py                ← Dokuman → chunk → embed → SQLite
├── retrieval.py             ← Sorgu → top-K chunk bul
├── generate.py              ← Context + soru → LLM → cevap (local/cloud)
├── config.py                ← MODE (local/cloud), model isimleri, sabitler
├── knowledge.db             ← SQLite veritabani (olusturulacak)
├── .streamlit/
│   └── secrets.toml         ← Streamlit Cloud icin (API key)
├── docs/                    ← Bilgi tabani dokumanlari (13 dosya)
│   ├── transfer_process.txt
│   ├── tuition_fees.txt
│   ├── course_registration.txt
│   ├── credit_transfer.txt
│   ├── software_engineering.txt
│   ├── visa_immigration.txt
│   ├── bearkat_life.txt
│   ├── orientation_arrival.txt
│   ├── financial_guide.txt
│   ├── oncampus_jobs.txt
│   ├── scholarships_financial_aid.txt
│   ├── offcampus_housing.txt
│   ├── faq_practical.txt
│   └── shsu_offcampus_housing_2026.pdf  ← Resmi kaynak PDF
├── teaching/                ← Ogretim materyalleri
│   ├── week1_exercises.md
│   ├── week2_exercises.md
│   ├── setup_guide_macos.md
│   ├── setup_guide_windows.md
│   └── demo_scenarios.md
└── app.py                   ← Streamlit web arayuzu (ana UI)
```

---

## Riskler ve Yedek Planlar

| Risk | Etki | Cozum |
|------|------|-------|
| Foundry Local sorun cikarabilir | Proje durur | **Ollama'ya gec** (yedek plan onaylandi) |
| Model indirme cok uzun surebilir | Zaman kaybi | Kucuk model sec, ilk is olarak baslat |
| Python 3.9 SDK ile uyumsuz olabilir | Kurulum hatasi | Python 3.12 yukle |
| 16GB RAM yetmeyebilir (buyuk model) | Yavas/crash | Phi-3.5 Mini (~2-4GB) ile kal |
| Embedding kalitesi dusuk olabilir | Yanlis chunk donuyor | Chunk boyutunu ayarla, top-K artir |
| Windows'ta path/encoding farki | Hata | `pathlib.Path` + UTF-8 encoding zorunlu |
| Turkce cevap kalitesi dusuk olabilir | Kotu UX | System prompt'u iyilestir, ornek cevaplar ekle |

### Ollama Yedek Plani

Foundry Local calismazsa:
```bash
# Ollama kurulumu
# macOS: brew install ollama
# Windows: ollama.com'dan installer indir
ollama pull phi3.5
ollama pull nomic-embed-text  # embedding modeli

# Kod degisikligi: sadece config.py'de runtime degistir
# RAG pipeline'in geri kalani ayni kalir
```
