# 🩺 Medical Chatbot - RAG System

A local, offline medical chatbot that answers questions based on the **Gale Encyclopedia of Medicine, 2nd Edition** using Retrieval-Augmented Generation (RAG).

## ✨ Features

- **100% Offline**: No external API calls, runs entirely locally
- **Source Citations**: Every answer includes page references from the medical encyclopedia
- **Professional Interface**: Clean Streamlit web UI with chat functionality
- **Medical Disclaimer**: Built-in safety warnings for users
- **Grounded Responses**: Answers only from the provided medical content

## 🛠️ Tech Stack

- **LLM**: TinyLlama (via Ollama)
- **Embeddings**: all-minilm (via Ollama)
- **Vector Database**: FAISS (local)
- **Framework**: LangChain
- **UI**: Streamlit
- **PDF Processing**: PyPDFLoader

## 📋 Prerequisites

1. **Python 3.8+**
2. **Ollama**: Download from [ollama.ai](https://ollama.ai)

## 🚀 Quick Start

### Step 1: Install Ollama and Models
```bash
# Download and install Ollama first, then:
ollama serve
ollama pull tinyllama
ollama pull all-minilm
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Build Vector Database
```bash
python vectorstore_optimized.py
```
This will:
- Load `data/Medical_book.pdf`
- Split into 500-character chunks with 100-character overlap
- Generate embeddings using `all-minilm`
- Save FAISS index to `vectorstore/faiss_medical_db/`

### Step 4: Launch the Web App
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 📁 Project Structure

```
medical-chatbot/
├── data/
│   └── Medical_book.pdf          # Source medical encyclopedia
├── vectorstore/
│   └── faiss_medical_db/         # FAISS vector database (auto-generated)
├── load_pdf.py                   # PDF loading and chunking
├── vectorstore.py                # Vector database creation
├── rag_chain.py                  # RAG pipeline implementation
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 💡 Usage Examples

Try these sample questions:

- "What is diabetes?"
- "What are the symptoms of hypertension?"
- "How is pneumonia treated?"
- "What causes heart disease?"
- "Tell me about migraine headaches"

## 🔧 Troubleshooting

### Ollama Connection Issues
```bash
# Make sure Ollama is running
ollama serve

# Check if models are available
ollama list
```

### Vector Store Issues
```bash
# Rebuild the vector database
python vectorstore.py
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

## ⚠️ Important Disclaimers

**This is for educational purposes only:**
- Not a substitute for professional medical advice
- Always consult qualified healthcare providers
- In emergencies, contact emergency services immediately
- Information based on encyclopedia content, not current medical practice

## 🏗️ System Architecture

1. **PDF Processing**: `load_pdf.py` splits the medical PDF into chunks
2. **Vector Storage**: `vectorstore.py` creates embeddings and FAISS index
3. **RAG Pipeline**: `rag_chain.py` handles retrieval and generation
4. **Web Interface**: `app.py` provides the chat UI

## 📊 Performance Notes

- **Vector DB Size**: ~50-100MB (depends on PDF size)
- **Query Time**: 2-5 seconds per question
- **Memory Usage**: ~1-2GB RAM
- **Models Size**: 
  - TinyLlama: ~637MB
  - all-minilm: ~23MB

## 🔒 Privacy & Security

- All processing happens locally
- No data sent to external servers
- Vector embeddings stored on local disk
- Chat history not persisted (resets on page refresh)

## 🛡️ License & Usage

This project is for educational and research purposes. The medical content belongs to Gale Encyclopedia of Medicine. Please respect copyright and use responsibly.
