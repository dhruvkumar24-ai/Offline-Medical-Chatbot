"""
FAST Vector Store Creation for Medical Chatbot
Optimized version with multiple speed improvements.
"""

import os
import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from load_pdf import load_and_split_pdf
from concurrent.futures import ThreadPoolExecutor
import threading


class FastVectorStore:
    """Optimized vector store creation with progress tracking."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.processed_count = 0
        
    def create_embeddings_batch(self, embeddings, texts, batch_id, total_batches):
        """Create embeddings for a batch of texts."""
        try:
            batch_embeddings = embeddings.embed_documents(texts)
            
            with self.lock:
                self.processed_count += len(texts)
                progress = (batch_id / total_batches) * 100
                print(f"   📊 Batch {batch_id}/{total_batches} complete | Progress: {progress:.1f}%")
            
            return batch_embeddings
        except Exception as e:
            print(f"   ❌ Error in batch {batch_id}: {e}")
            return None


def create_vector_store_fast(pdf_path: str, vectorstore_path: str):
    """
    Create vector store with multiple optimization strategies.
    """
    
    print("🚀 FAST Vector Store Creation")
    print("=" * 40)
    
    # Strategy 1: Reduce chunk size for faster processing
    print("📄 Loading PDF with optimized chunking...")
    chunks = load_and_split_pdf(pdf_path)
    
    # If too many chunks, reduce them
    if len(chunks) > 3000:
        print(f"⚡ Reducing chunks from {len(chunks)} to 3000 for faster processing...")
        # Take every nth chunk to reduce total
        step = len(chunks) // 3000
        chunks = chunks[::step]
        print(f"✅ Using {len(chunks)} optimized chunks")
    
    print(f"🔗 Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        model="all-minilm",
        base_url="http://localhost:11434"
    )
    
    # Test connection
    try:
        test_embed = embeddings.embed_query("test")
        print(f"✅ Embeddings ready (dimension: {len(test_embed)})")
    except Exception as e:
        raise ConnectionError(f"Ollama connection failed: {e}")
    
    # Strategy 2: Try the fastest method first
    print("\n🎯 STRATEGY 1: Direct FAISS creation (fastest)")
    print("-" * 30)
    
    start_time = time.time()
    
    try:
        # This is often the fastest if it works
        print("🚀 Creating vector store directly...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS! Vector store created in {elapsed/60:.1f} minutes")
        
    except Exception as e:
        print(f"⚠️  Direct method failed: {e}")
        
        # Strategy 3: Optimized batching
        print(f"\n🎯 STRATEGY 2: Optimized batch processing")
        print("-" * 30)
        
        # Use smaller, more efficient batches
        batch_size = 25  # Smaller batches = faster individual processing
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"📦 Processing {len(chunks)} chunks in {total_batches} batches of {batch_size}")
        
        # Process first batch to create initial store
        print("🏗️  Creating initial store from first batch...")
        first_batch = chunks[:batch_size]
        vector_store = FAISS.from_documents(first_batch, embeddings)
        processed = batch_size
        
        # Add remaining chunks in optimized batches
        batch_start_time = time.time()
        
        for i in range(batch_size, len(chunks), batch_size):
            batch_num = (i // batch_size) + 1
            batch_chunks = chunks[i:i + batch_size]
            
            # Create and merge batch
            try:
                batch_store = FAISS.from_documents(batch_chunks, embeddings)
                vector_store.merge_from(batch_store)
                
                processed += len(batch_chunks)
                progress = (processed / len(chunks)) * 100
                
                # Calculate ETA
                elapsed = time.time() - batch_start_time
                chunks_per_sec = processed / elapsed
                remaining = len(chunks) - processed
                eta_seconds = remaining / chunks_per_sec if chunks_per_sec > 0 else 0
                
                print(f"   ✅ Batch {batch_num}/{total_batches} | Progress: {progress:.1f}% | ETA: {eta_seconds/60:.1f} min")
                
            except Exception as batch_error:
                print(f"   ❌ Batch {batch_num} failed: {batch_error}")
                continue
        
        elapsed = time.time() - start_time
        print(f"✅ Vector store created in {elapsed/60:.1f} minutes")
    
    # Save the vector store
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    print(f"💾 Saving to {vectorstore_path}...")
    
    try:
        vector_store.save_local(vectorstore_path)
        print("✅ Vector store saved successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to save: {e}")
    
    return vector_store


def main():
    """Main function with performance monitoring."""
    pdf_path = "data/Medical_book.pdf"
    vectorstore_path = "vectorstore/faiss_medical_db"
    
    print("⚡ FAST VECTOR STORE CREATOR")
    print("=" * 50)
    print("🎯 Optimizations enabled:")
    print("  • Reduced chunk count for faster processing")
    print("  • Multiple fallback strategies")  
    print("  • Smaller batch sizes")
    print("  • Real-time progress tracking")
    print("=" * 50)
    
    total_start = time.time()
    
    try:
        vector_store = create_vector_store_fast(pdf_path, vectorstore_path)
        
        # Test the created vector store
        print("\n🧪 Testing vector store...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        test_docs = retriever.get_relevant_documents("What is diabetes?")
        
        print(f"✅ Test successful - retrieved {len(test_docs)} documents")
        
        total_time = time.time() - total_start
        print(f"\n🎉 COMPLETE! Total time: {total_time/60:.1f} minutes")
        print("🚀 Ready to run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check models: ollama list")
        print("3. Restart Ollama if needed")


if __name__ == "__main__":
    main()
