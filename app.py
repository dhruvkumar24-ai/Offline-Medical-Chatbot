"""
Streamlit Web Application for Medical Chatbot
User-friendly chat interface for the medical RAG system.
"""

import streamlit as st
import os
from rag_chain import MedicalRAGChain


# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability and high contrast
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #ffffff;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

.disclaimer-box {
    background-color: #fff3cd;
    border: 3px solid #ff6b35;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    color: #8b0000;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.chat-message {
    padding: 1.2rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border: 2px solid #ddd;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.user-message {
    background-color: #e8f4fd;
    border-left: 6px solid #0066cc;
    color: #003d7a;
    font-weight: 500;
}

.assistant-message {
    background-color: #f0f8ff;
    border-left: 6px solid #28a745;
    color: #1a5330;
    font-weight: 500;
}

.source-box {
    background-color: #ffffff;
    border: 2px solid #0066cc;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 1rem;
    color: #333333;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.source-box strong {
    color: #0066cc;
    font-size: 1.1rem;
    display: block;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


def load_rag_chain():
    """Load the RAG chain if not already loaded."""
    if st.session_state.rag_chain is None:
        vectorstore_path = "vectorstore/faiss_medical_db"
        
        if not os.path.exists(vectorstore_path):
            st.error("❌ Vector store not found! Please run: `python vectorstore.py`")
            st.stop()
        
        try:
            with st.spinner("Loading medical knowledge base..."):
                st.session_state.rag_chain = MedicalRAGChain(vectorstore_path)
        except Exception as e:
            st.error(f"❌ Failed to load RAG chain: {e}")
            st.error("Make sure Ollama is running: `ollama serve`")
            st.stop()


def display_disclaimer():
    """Display medical disclaimer with high contrast."""
    st.markdown("""
    <div class="disclaimer-box">
        <h4 style="color: #8b0000; margin-bottom: 1rem;">⚠️ Important Medical Disclaimer</h4>
        <p style="color: #8b0000; font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">
            This chatbot is for educational and informational purposes only.
        </p>
        <ul style="color: #8b0000; font-size: 1rem; line-height: 1.6;">
            <li><strong>This is NOT a substitute for professional medical advice, diagnosis, or treatment</strong></li>
            <li><strong>Always consult a qualified healthcare provider for medical concerns</strong></li>
            <li><strong>In case of medical emergency, call emergency services immediately</strong></li>
            <li><strong>Information is based on the Gale Encyclopedia of Medicine, 2nd Edition</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def display_chat_message(role: str, content: str, sources=None):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🙋‍♂️ You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🩺 Medical Assistant:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if sources and len(sources) > 0:
            with st.expander(f"📚 View Sources ({len(sources)} references)", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i} (Page {source['page']}):</strong>
                        <p style="color: #333333; margin-top: 0.5rem; line-height: 1.5;">
                            {source['content'][:300]}{'...' if len(source['content']) > 300 else ''}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header">🩺 Medical Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #333333; font-size: 1.2rem; font-weight: 500;">Ask questions about medical conditions, symptoms, and treatments</p>', unsafe_allow_html=True)
    
    # Display disclaimer
    display_disclaimer()
    
    # Load RAG chain
    load_rag_chain()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This chatbot answers medical questions using information from the 
        **Gale Encyclopedia of Medicine, 2nd Edition**.
        
        **Features:**
        - Offline operation (no internet required)
        - Source citations for transparency
        - Professional medical language
        
        **Tips for better results:**
        - Be specific in your questions
        - Ask about symptoms, conditions, or treatments
        - Check the sources for detailed information
        """)
        
        st.header("🔧 System Status")
        if st.session_state.rag_chain:
            st.success("✅ RAG System: Active")
            st.success("✅ Vector DB: Loaded")
            st.success("✅ Ollama: Connected")
        else:
            st.warning("⚠️ System: Not Ready")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.experimental_rerun()
    
    # Chat interface
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question (e.g., 'What is diabetes?')"):
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Get AI response
        with st.spinner("Searching medical knowledge base..."):
            try:
                result = st.session_state.rag_chain.ask_question(prompt)
                
                if result["success"]:
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    # Add disclaimer to answer
                    answer_with_disclaimer = f"{answer}\n\n*Remember: This information is for educational purposes only. Consult a healthcare provider for medical advice.*"
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_with_disclaimer,
                        "sources": sources
                    })
                    
                    # Display the response
                    display_chat_message("assistant", answer_with_disclaimer, sources)
                    
                else:
                    error_message = "Sorry, I encountered an error processing your question. Please try again or rephrase your question."
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    display_chat_message("assistant", error_message)
                    
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>Powered by 🦙 Ollama • 🔗 LangChain • 🗃️ FAISS • 🎈 Streamlit</p>
        <p>Medical content from <em>Gale Encyclopedia of Medicine, 2nd Edition</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
