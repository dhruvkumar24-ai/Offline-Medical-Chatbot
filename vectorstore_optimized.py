"""
Optimized Vector Store Creation - All Chunks with Speed Improvements
Uses all chunks but with multiple speed optimizations for embeddings.
"""

import os
import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from load_pdf import load_and_split_pdf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple


class OptimizedEmbeddings:
    """Optimized embedding class with batch processing and threading."""
    
    def __init__(self, model: str = "all-minilm", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()  # Reuse connections
        self.lock = threading.Lock()
        self.processed = 0
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    
    def embed_documents_batch(self, texts: List[str], batch_id: int = 0) -> List[List[float]]:
        """Embed a batch of documents with optimizations."""
        embeddings = []
        
        for text in texts:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
                
                with self.lock:
                    self.processed += 1
                    
            except Exception as e:
                print(f"   ⚠️  Error in batch {batch_id}, text {len(embeddings)+1}: {e}")
                # Use a zero vector as fallback
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    # Get dimension from a test embedding
                    try:
                        test_emb = self.embed_query("test")
                        embeddings.append([0.0] * len(test_emb))
                    except:
                        embeddings.append([0.0] * 384)  # Default dimension
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with parallel processing."""
        if len(texts) <= 50:
            # Small batch, process directly
            return self.embed_documents_batch(texts)
        
        # Large batch, use parallel processing
        batch_size = 25  # Smaller batches for better error handling
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        all_embeddings = []
        
        # Process batches with threading for speed
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_batch = {
                executor.submit(self.embed_documents_batch, batch, i): i 
                for i, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_embeddings = future.result()
                    # Insert at correct position to maintain order
                    while len(all_embeddings) <= batch_id:
                        all_embeddings.append([])
                    all_embeddings[batch_id] = batch_embeddings
                    
                    progress = (self.processed / len(texts)) * 100
                    print(f"   📊 Progress: {progress:.1f}% ({self.processed}/{len(texts)} processed)")
                    
                except Exception as e:
                    print(f"   ❌ Batch {batch_id} failed: {e}")
        
        # Flatten results while maintaining order
        result = []
        for batch_embeddings in all_embeddings:
            result.extend(batch_embeddings)
        
        return result


def create_optimized_vector_store(pdf_path: str, vectorstore_path: str):
    """
    Create vector store with all chunks but optimized processing.
    """
    
    print("⚡ OPTIMIZED Vector Store Creation")
    print("🎯 Using ALL chunks with speed optimizations")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load PDF with your original chunk settings
    print("📄 Loading PDF...")
    chunks = load_and_split_pdf(pdf_path)
    print(f"✅ Loaded {len(chunks)} chunks (keeping all chunks as requested)")
    
    # Use optimized embeddings
    print("🔗 Initializing optimized embeddings...")
    embeddings = OptimizedEmbeddings(model="all-minilm")
    
    # Test connection
    try:
        test_embed = embeddings.embed_query("test")
        print(f"✅ Embeddings ready (dimension: {len(test_embed)})")
    except Exception as e:
        raise ConnectionError(f"Ollama connection failed: {e}")
    
    # Create vector store with optimized processing
    print(f"🚀 Creating vector store from {len(chunks)} chunks...")
    print("💡 Optimizations enabled:")
    print("   • Connection pooling for faster requests")
    print("   • Parallel batch processing")
    print("   • Error recovery with fallbacks")
    print("   • Real-time progress tracking")
    print("-" * 30)
    
    creation_start = time.time()
    
    try:
        # Extract texts for embedding
        texts = [chunk.page_content for chunk in chunks]
        
        # Create embeddings with progress tracking
        print("🔄 Generating embeddings...")
        all_embeddings = embeddings.embed_documents(texts)
        
        # Create FAISS index
        print("🏗️  Building FAISS index...")
        import numpy as np
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Use LangChain's FAISS wrapper for compatibility
        standard_embeddings = OllamaEmbeddings(model="all-minilm")
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, all_embeddings)),
            embedding=standard_embeddings
        )
        
        creation_time = time.time() - creation_start
        print(f"✅ Vector store created in {creation_time/60:.1f} minutes!")
        
    except Exception as e:
        print(f"⚠️  Optimized method failed: {e}")
        print("🔄 Falling back to standard method...")
        
        # Fallback to standard LangChain method
        standard_embeddings = OllamaEmbeddings(model="all-minilm")
        vector_store = FAISS.from_documents(chunks, standard_embeddings)
        
        creation_time = time.time() - creation_start
        print(f"✅ Vector store created with fallback in {creation_time/60:.1f} minutes")
    
    # Save vector store
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    print(f"💾 Saving to {vectorstore_path}...")
    vector_store.save_local(vectorstore_path)
    
    # Test retrieval
    print("🧪 Testing retrieval...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    test_docs = retriever.get_relevant_documents("What is diabetes?")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("🎉 SUCCESS!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Total chunks processed: {len(chunks)}")
    print(f"🧪 Test query returned: {len(test_docs)} documents")
    print("🚀 Ready to run: streamlit run app.py")
    print("=" * 50)
    
    return vector_store


def main():
    """Main function."""
    pdf_path = "data/Medical_book.pdf"
    vectorstore_path = "vectorstore/faiss_medical_db"
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
        else:
            raise ConnectionError("Ollama not responding")
    except:
        print("❌ Ollama is not running!")
        print("Please start Ollama first: ollama serve")
        return
    
    try:
        create_optimized_vector_store(pdf_path, vectorstore_path)
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check models: ollama list")
        print("3. Restart Ollama if needed")


if __name__ == "__main__":
    main()
