"""
Vector Store Creation for Medical Chatbot
Creates and saves a FAISS vector database from medical PDF chunks.
"""

import os
import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from load_pdf import load_and_split_pdf


def create_vector_store(pdf_path: str, vectorstore_path: str):
    """
    Create and save FAISS vector store from PDF chunks.
    
    Args:
        pdf_path (str): Path to the medical PDF
        vectorstore_path (str): Path to save the vector store
    """
    
    print("🔄 Starting vector store creation...")
    
    # Load and split PDF into chunks
    print("📄 Loading PDF...")
    chunks = load_and_split_pdf(pdf_path)
    
    if not chunks:
        raise ValueError("No chunks were created from the PDF")
    
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Initialize Ollama embeddings
    print("🔗 Connecting to Ollama embeddings...")
    try:
        embeddings = OllamaEmbeddings(
            model="all-minilm",
            base_url="http://localhost:11434"
        )
        
        # Test embedding connection
        test_embed = embeddings.embed_query("test")
        print(f"✅ Embeddings working (dimension: {len(test_embed)})")
        
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Ollama embeddings: {e}")
    
    # Create FAISS vector store with progress tracking
    print("🗃️ Creating FAISS vector store...")
    print(f"📊 Processing {len(chunks)} chunks (this will take several minutes)...")
    
    start_time = time.time()
    
    try:
        # Process chunks in batches to show progress
        batch_size = 50
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"🔄 Processing in {total_batches} batches of {batch_size} chunks each...")
        
        # Create vector store from first batch
        first_batch = chunks[:batch_size]
        batch_start = time.time()
        print(f"📝 Processing batch 1/{total_batches} ({len(first_batch)} chunks)...")
        vector_store = FAISS.from_documents(first_batch, embeddings)
        batch_time = time.time() - batch_start
        
        print(f"   ⏱️  Batch completed in {batch_time:.1f} seconds")
        
        # Estimate total time
        estimated_total = batch_time * total_batches
        print(f"   🕐 Estimated total time: {estimated_total/60:.1f} minutes")
        
        # Add remaining chunks in batches
        for i in range(1, total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(chunks))
            batch = chunks[start_idx:end_idx]
            
            batch_start = time.time()
            print(f"📝 Processing batch {i+1}/{total_batches} ({len(batch)} chunks)...")
            
            # Create temporary vector store for this batch
            batch_vs = FAISS.from_documents(batch, embeddings)
            
            # Merge with main vector store
            vector_store.merge_from(batch_vs)
            
            batch_time = time.time() - batch_start
            elapsed = time.time() - start_time
            
            # Show progress and time estimates
            progress = ((i + 1) / total_batches) * 100
            avg_batch_time = elapsed / (i + 1)
            remaining_batches = total_batches - (i + 1)
            eta = remaining_batches * avg_batch_time
            
            print(f"   ✅ Progress: {progress:.1f}% | Batch time: {batch_time:.1f}s | ETA: {eta/60:.1f} min")
        
        total_time = time.time() - start_time
        print(f"✅ Vector store created successfully in {total_time/60:.1f} minutes!")
        
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
    
    # Save vector store
    print(f"💾 Saving vector store to: {vectorstore_path}")
    try:
        vector_store.save_local(vectorstore_path)
        print("✅ Vector store saved successfully!")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save vector store: {e}")
    
    return vector_store


def load_vector_store(vectorstore_path: str):
    """
    Load existing FAISS vector store.
    
    Args:
        vectorstore_path (str): Path to the saved vector store
        
    Returns:
        FAISS: Loaded vector store
    """
    
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vector store not found: {vectorstore_path}")
    
    print(f"📂 Loading vector store from: {vectorstore_path}")
    
    try:
        embeddings = OllamaEmbeddings(
            model="all-minilm",
            base_url="http://localhost:11434"
        )
        
        vector_store = FAISS.load_local(
            vectorstore_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("✅ Vector store loaded successfully")
        return vector_store
        
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")


def main():
    """
    Main function to create the vector store.
    """
    pdf_path = "data/Medical_book.pdf"
    vectorstore_path = "vectorstore/faiss_medical_db"
    
    try:
        # Check if Ollama is running
        print("🔍 Checking Ollama connection...")
        
        # Create vector store
        vector_store = create_vector_store(pdf_path, vectorstore_path)
        
        # Test retrieval
        print("\n🧪 Testing retrieval...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        test_docs = retriever.get_relevant_documents("What is diabetes?")
        
        print(f"✅ Retrieved {len(test_docs)} documents for test query")
        for i, doc in enumerate(test_docs):
            print(f"  Doc {i+1}: {doc.page_content[:100]}...")
        
        print("\n🎉 Vector store creation completed successfully!")
        print("   You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure models are pulled:")
        print("   - ollama pull tinyllama")
        print("   - ollama pull all-minilm")
        print("3. Check if Medical_book.pdf exists in data/ folder")


if __name__ == "__main__":
    main()
