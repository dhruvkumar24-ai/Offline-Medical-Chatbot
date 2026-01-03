"""
Setup and Test Script for Medical Chatbot
Run this to verify everything is working correctly.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_ollama_service():
    """Check if Ollama service is running."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Ollama service running (version: {version_info.get('version', 'unknown')})")
            return True
    except:
        pass
    
    print("❌ Ollama service not running")
    print("   Start with: ollama serve")
    return False


def check_ollama_models():
    """Check if required models are available."""
    try:
        # Check tinyllama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "tinyllama", "prompt": "test", "stream": False},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ tinyllama model available")
            tinyllama_ok = True
        else:
            print("❌ tinyllama model not found")
            print("   Install with: ollama pull tinyllama")
            tinyllama_ok = False
    except:
        print("❌ tinyllama model not available")
        tinyllama_ok = False
    
    try:
        # Check all-minilm (embeddings)
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "all-minilm", "prompt": "test"},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ all-minilm embedding model available")
            embeddings_ok = True
        else:
            print("❌ all-minilm model not found")
            print("   Install with: ollama pull all-minilm")
            embeddings_ok = False
    except:
        print("❌ all-minilm model not available")
        embeddings_ok = False
    
    return tinyllama_ok and embeddings_ok


def check_pdf_file():
    """Check if the medical PDF exists."""
    pdf_path = Path("data/Medical_book.pdf")
    if pdf_path.exists():
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"✅ Medical PDF found ({size_mb:.1f} MB)")
        return True
    else:
        print("❌ Medical PDF not found at data/Medical_book.pdf")
        return False


def check_dependencies():
    """Check if Python packages are installed."""
    required_packages = [
        "langchain",
        "langchain_community",
        "faiss_cpu",
        "pypdf",
        "streamlit",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_vector_store():
    """Check if vector store exists."""
    vs_path = Path("vectorstore/faiss_medical_db")
    if vs_path.exists():
        print("✅ Vector store exists")
        return True
    else:
        print("❌ Vector store not found")
        print("   Create with: python vectorstore.py")
        return False


def run_setup_check():
    """Run complete setup check."""
    print("🔍 Medical Chatbot Setup Check")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Ollama Service", check_ollama_service),
        ("Ollama Models", check_ollama_models),
        ("Medical PDF", check_pdf_file),
        ("Python Dependencies", check_dependencies),
        ("Vector Store", check_vector_store)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔸 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    
    if all_passed:
        print("🎉 All checks passed! Ready to run the chatbot.")
        print("   Start with: streamlit run app.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    
    return all_passed


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def main():
    """Main setup function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_dependencies()
    
    run_setup_check()


if __name__ == "__main__":
    main()
