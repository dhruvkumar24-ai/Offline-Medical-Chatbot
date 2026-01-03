"""
PDF Document Loader for Medical Chatbot
Loads and splits the medical PDF into chunks for RAG processing.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document


def load_and_split_pdf(pdf_path: str) -> List[Document]:
    """
    Load PDF and split into chunks suitable for RAG.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Document]: List of document chunks with metadata
    """
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Loading PDF from: {pdf_path}")
    
    # Load PDF using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages from PDF")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks")
    
    # Add chunk information to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "source_file": os.path.basename(pdf_path)
        })
    
    return chunks


def main():
    """
    Test function to load and display PDF chunks.
    """
    pdf_path = "data/Medical_book.pdf"
    
    try:
        chunks = load_and_split_pdf(pdf_path)
        
        print("\n" + "="*50)
        print("SAMPLE CHUNKS:")
        print("="*50)
        
        # Display first 3 chunks as samples
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- CHUNK {i+1} ---")
            print(f"Page: {chunk.metadata.get('page', 'Unknown')}")
            print(f"Content: {chunk.page_content[:200]}...")
            print(f"Metadata: {chunk.metadata}")
        
        print(f"\nTotal chunks created: {len(chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
