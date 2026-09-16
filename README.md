---

<h1 align="center">🏥 (Surgical Procedures RAG)</h1>
<h3 align="center">Advanced Persian Surgical RAG System</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-🦜🔗-green?style=flat-square" alt="LangChain">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-orange?style=flat-square" alt="FAISS">
  <img src="https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenRouter-LLM_API-purple?style=flat-square" alt="OpenRouter">
</p>

<p align="center">
  <a href="#-architecture">Architecture</a> •
  <a href="#-key-engineering-challenges-solved">Challenges Solved</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation--setup">Setup</a>
</p>

---



## 📖 Overview

**JarahYar (جراح‌یار)** is a production-grade, multi-tenant Retrieval-Augmented Generation (RAG) system specifically engineered for **Persian medical and surgical protocols**.

Standard RAG pipelines fail on Persian medical texts due to morphological variations, RTL rendering bugs, and massive cross-document redundancy. This system solves these issues at the ingestion, retrieval, and presentation layers, providing operating room staff and medical students with a highly accurate, hallucination-free surgical assistant.

![UI Screenshot Placeholder](https://github.com/the-voivode/Medical-RAG-system-for-surgical-procedures/blob/6897fd601ae6618d71505d4e8f536b858035dd30/assets/images/Screenshot%20(35).png)
---

## 🚀 Key Engineering Challenges Solved

### 1. The "Redundant Protocol" Problem (Context Deduplication)

Surgical manuals contain massive redundancy (e.g., the exact same "Cautery Modes" or "Sponge Counting" section appears across 15 different surgery documents). Standard RAG feeds all 15 copies to the LLM, wasting tokens and causing hallucinations.

* **The Fix:** Implemented a retrieval-time `SequenceMatcher` deduplication layer. It collapses identical cross-document sections into a single "Shared Note" while preserving the list of source documents, **saving up to 60% of LLM context tokens** and improving answer diversity.

### 2. Persian Morphology & Keyword Blindness

Persian text suffers from character variations (ي vs ی, ك vs ک) and ZWNJ (نیم‌فاصله) inconsistencies. Furthermore, BM25 keyword search often fails if the exact keyword is in the section title rather than the chunk body.

* **The Fix:**
  * **Ingestion:** `hazm`-based structural normalization that preserves RAG delimiters while unifying characters.
  * **Retrieval:** BM25 is explicitly configured to index the `page_content` (which includes the prepended Document Name and Section Title), ensuring keyword queries like *"Appendectomy Indications"* actually hit the correct sections.

### 3. Multi-Tenant API Isolation

Running a multi-user LLM app usually results in shared rate limits and billing nightmares.

* **The Fix:** Built a custom authentication and routing layer where **each user's session is tied to their own OpenRouter API key**. The heavy RAG pipeline (FAISS/BM25) is cached globally in RAM via `@st.cache_resource`, but LLM generation requests are isolated per-user.

---

## 🏗️ Architecture

```text
[Raw .md Files]
      │
      ▼
[1. Hazm Normalizer] ──► Fixes Persian morphology, preserves ===PARENT=== / ===CHILD=== markers
      │
      ▼
[2. Parent-Child Chunker] ──► Small chunks for precise retrieval, full parents for rich LLM context
      │
      ├─► [FAISS Index] (Dense: BAAI/bge-m3)
      └─► [BM25 Index] (Sparse: Persian tokenization + Stopwords)
      │
      ▼
[3. Hybrid Retriever] ──► Reciprocal Rank Fusion (RRF) + Soft Consensus Gate
      │
      ▼
[4. Post-Processing] ──► Parent Expansion + Cross-Doc Deduplication + Image Metadata Extraction
      │
      ▼
[5. OpenRouter LLM] ──► Strict Medical Grounding Prompt + RTL Formatting Rules
      │
      ▼
[6. Streamlit UI] ──► Vazirmatn Font + RTL CSS Injection + Streaming Output
````

---

## 🛠️ Tech Stack

* **Core AI:** LangChain, OpenAI SDK (via OpenRouter)
* **Vector & Keyword Search:** FAISS, `rank_bm25`
* **NLP / Persian Processing:** `hazm` (Normalization, Tokenization, Stopwords)
* **Embeddings:** `BAAI/bge-m3` (State-of-the-art multilingual dense retriever)
* **Backend / Auth:** `pydantic-settings`, `bcrypt`, Custom JSON-based Chat Persistence
* **Frontend:** Streamlit (with custom CSS for RTL and Persian typography)

---

## 💻 Installation & Setup

### 1. Clone and Install

```bash
git clone https://github.com/the-voivode/Medical-RAG-system-for-surgical-procedures.git
cd Medical-RAG-system-for-surgical-procedures
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
# Admin Controls (Global settings for all users)
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
USE_CHAT_MEMORY=true
MEMORY_MESSAGES=3

# Hardware
DEVICE=cpu  # Change to 'cuda' if you have an NVIDIA GPU
```

### 3. Data Preparation Pipeline

Place your raw `.md` surgical protocols in the `docs/` folder, then run the pipeline:

```bash
# 1. Normalize Persian text
python normalizer.py

# 2. Chunk into Parents and Children
python chunker.py
```

### 4. Create an Admin User

Users require an OpenRouter API key to generate answers. Create your first user via the CLI:

```bash
python create_user.py --username admin --password securepass --api-key sk-or-v1-your-key-here
```

### 5. Run the Application

```bash
streamlit run app.py
```

*(Note: The FAISS and BM25 indexes will automatically build on the first run and cache to disk).*

---

## 🧠 Advanced Prompt Engineering

The system uses a highly constrained system prompt designed specifically for medical accuracy:

1. **Strict Grounding:** Forces the LLM to reply with a specific Persian fallback phrase if the context lacks the answer, preventing medical hallucinations.
2. **Terminology Preservation:** Instructs the LLM to keep English medical device modes (e.g., *SWIFT COAG*, *LigaSure*) intact while answering in Persian.
3. **RTL Formatting Hacks:** Bypasses Streamlit's Markdown indentation bugs in RTL mode by forcing the LLM to use Persian digits (`۱)`, `۲)`) instead of ASCII Markdown lists.
4. **Trigger-Based Structuring:** If the user asks about "ابزار" (Tools), the prompt dynamically forces the LLM to map tools to specific surgical steps and append safety warnings.

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit UI, Auth gate, Chat routing
├── auth.py                # bcrypt password hashing & user management
├── chatstore.py           # Per-user persistent JSON chat history
├── config.py              # Pydantic settings management
├── normalizer.py          # hazm-based structural text normalization
├── chunker.py             # Parent-Child markdown parser & chunker
├── embedder.py            # FAISS + BM25 Hybrid Retriever & RRF logic
├── generator.py           # LangChain/OpenRouter LLM orchestration
├── bm25builder.py         # Persian-aware BM25 wrapper
├── docs/                  # Raw surgical .md protocols
├── docs_normalized/       # Output of normalizer.py
├── chunked/               # Generated parents.json & children.json
├── faiss_index/           # Cached dense vector store
└── user_data/             # Isolated per-user chat histories
```



<p align="center">
  <i>Built with precision for the Persian medical community.</i>
</p>
```
