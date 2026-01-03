"""
RAG Chain for Medical Chatbot
Builds the complete RAG pipeline with prompt engineering.
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Dict, List
import os


class MedicalRAGChain:
    """
    Medical RAG Chain that answers questions based on medical PDF content.
    """
    
    def __init__(self, vectorstore_path: str):
        """
        Initialize the RAG chain.
        
        Args:
            vectorstore_path (str): Path to the FAISS vector store
        """
        self.vectorstore_path = vectorstore_path
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the complete RAG chain."""
        
        # Initialize LLM
        print("🤖 Initializing Ollama LLM...")
        self.llm = Ollama(
            model="tinyllama",
            base_url="http://localhost:11434",
            temperature= 0.2
        )
        
        # Load vector store
        print("📂 Loading vector store...")
        embeddings = OllamaEmbeddings(
            model="all-minilm",
            base_url="http://localhost:11434"
        )
        
        self.vector_store = FAISS.load_local(
            self.vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Define prompt template
        prompt_template = """You are a medical assistant. Answer the question using only the information provided in the context below.

If the answer is not in the context, say "I don't know based on the provided information."

Be clear, concise, and professional. Always include relevant details from the context.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        print("✅ RAG chain initialized successfully")
    
    def ask_question(self, question: str) -> Dict:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question (str): The user's question
            
        Returns:
            Dict: Contains 'answer' and 'sources'
        """
        
        try:
            # Get response from QA chain
            result = self.qa_chain({"query": question})
            
            # Extract answer and source documents
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Process sources for display
            sources = []
            for doc in source_docs:
                source_info = {
                    "content": doc.page_content.strip(),
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Medical_book.pdf")
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def get_similar_content(self, question: str, k: int = 3) -> List[Dict]:
        """
        Get similar content chunks for a question (for debugging).
        
        Args:
            question (str): The query
            k (int): Number of similar chunks to retrieve
            
        Returns:
            List[Dict]: Similar content chunks
        """
        
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(question)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "Unknown"),
                    "score": getattr(doc, "score", None)
                })
            
            return results
            
        except Exception as e:
            print(f"Error retrieving similar content: {e}")
            return []


def test_rag_chain():
    """
    Test function for the RAG chain.
    """
    vectorstore_path = "vectorstore/faiss_medical_db"
    
    if not os.path.exists(vectorstore_path):
        print("❌ Vector store not found. Please run: python vectorstore.py")
        return
    
    try:
        print("🧪 Testing RAG Chain...")
        
        # Initialize RAG chain
        rag = MedicalRAGChain(vectorstore_path)
        
        # Test questions
        test_questions = [
            "What is diabetes?",
            "What are the symptoms of hypertension?",
            "How is pneumonia treated?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 50)
            
            result = rag.ask_question(question)
            
            if result["success"]:
                print(f"🤖 Answer: {result['answer']}")
                print(f"📚 Sources found: {len(result['sources'])}")
                
                for i, source in enumerate(result['sources']):
                    print(f"   Source {i+1} (Page {source['page']}): {source['content'][:100]}...")
            else:
                print(f"❌ Error: {result['answer']}")
        
        print("\n✅ RAG chain test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_rag_chain()
