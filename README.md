# 🩺 Offline Medical RAG Chatbot
### LangChain · TinyLlama · FAISS · Ollama · Streamlit · Self-Project

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-1C3C3C?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-TinyLlama-black?style=flat-square)
![Status](https://img.shields.io/badge/Status-Offline_%7C_No_API_Key_Needed-brightgreen?style=flat-square)

---

## 🧩 Problem Statement

Medical information is scattered across large encyclopedias — hard to search, harder to query conversationally. This project builds a **fully offline RAG-based medical chatbot** that answers natural language questions from an 800+ page medical PDF with source citations, zero external API calls, and a clean Streamlit interface.

> *"Ask a medical question → retrieve relevant chunks → generate grounded, cited answers — all locally on your machine."*

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **100% Offline** | No OpenAI/API key needed — runs entirely on local Ollama models |
| **Source Citations** | Every answer includes page references from the encyclopedia |
| **Grounded Responses** | Answers strictly from retrieved context — no hallucination |
| **Medical Disclaimer** | Built-in safety warnings for responsible use |
| **Clean Chat UI** | Streamlit-based interface with chat history |
| **Optimized Chunking** | 500-char chunks, 100-char overlap — 7,486 chunks from 800+ page PDF |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| **LLM** | TinyLlama (via Ollama — local) |
| **Embeddings** | all-MiniLM (via Ollama — local) |
| **Vector Database** | FAISS (local index) |
| **RAG Framework** | LangChain |
| **PDF Processing** | PyPDFLoader + RecursiveCharacterTextSplitter |
| **Web Interface** | Streamlit |

---

## ⚙️ System Architecture

```
Medical PDF (800+ pages)
        │
        ▼
PyPDFLoader → RecursiveCharacterTextSplitter
  (chunk_size=500, overlap=100) → 7,486 chunks
        │
        ▼
all-MiniLM Embeddings (Ollama)
        │
        ▼
FAISS Vector Index (local disk)
        │
        ▼
User Query → Similarity Search (top-k=3)
        │
        ▼
TinyLlama (Ollama) + Custom RAG Prompt
        │
        ▼
Grounded Answer + Source Citations
        │
        ▼
Streamlit Chat Interface
```

---

## 📁 Repository Structure

```
📦 Offline-Medical-RAG-Chatbot/
├── 📂 data/
│   └── Medical_book.pdf              # Source: Gale Encyclopedia of Medicine, 2nd Ed.
├── 📂 vectorstore/
│   └── faiss_medical_db/             # FAISS index (auto-generated)
├── load_pdf.py                       # PDF loading & chunking
├── vectorstore.py                    # Standard vector store creation
├── vectorstore_optimized.py          # Parallel batch embedding creation
├── vectorstore_fast.py               # Fast vector store with fallback strategies
├── rag_chain.py                      # RAG pipeline (retriever + LLM + prompt)
├── app.py                            # Streamlit chat interface
├── setup_check.py                    # Environment & dependency checker
├── setup.bat                         # Windows one-click setup
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Install Ollama & Pull Models
```bash
# Install Ollama from https://ollama.ai, then:
ollama serve
ollama pull tinyllama
ollama pull all-minilm
```

### Step 2 — Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Verify Setup
```bash
python setup_check.py
```

### Step 4 — Build Vector Database
```bash
python vectorstore_optimized.py
```
This processes `data/Medical_book.pdf` → generates embeddings → saves FAISS index.

### Step 5 — Launch App
```bash
streamlit run app.py
```
App runs at `http://localhost:8501`

---

## 💡 Sample Questions

```
"What is diabetes?"
"What are the symptoms of hypertension?"
"How is pneumonia treated?"
"What causes heart disease?"
"Tell me about migraine headaches"
```

---

## 📊 Performance

| Metric | Value |
|---|---|
| PDF Size | 800+ pages |
| Total Chunks | 7,486 |
| Chunk Size | 500 characters |
| Chunk Overlap | 100 characters |
| Query Response Time | 2–5 seconds |
| Memory Usage | ~1–2 GB RAM |
| TinyLlama Model Size | ~637 MB |
| all-MiniLM Model Size | ~23 MB |

---

## 🔒 Privacy

- All processing happens **locally** — no data sent to external servers
- No API keys required
- Chat history not persisted (resets on page refresh)

---

## ⚠️ Medical Disclaimer

This chatbot is for **educational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 🧠 Concepts Covered

`Retrieval-Augmented Generation (RAG)` · `Vector Embeddings` · `FAISS Indexing` · `LLM Inference` · `Prompt Engineering` · `PDF Processing` · `LangChain Pipelines` · `Streamlit UI`

---

## 👤 Author

**Dhruv Kumar Sahu**
M.Tech, Industrial & Management Engineering — IIT Kanpur
GATE 2024 AIR 33 | [LinkedIn](https://www.linkedin.com/in/dhruv-kumar-sahu-157ab9193/) · [GitHub](https://github.com/dhruvkumar24-ai)
