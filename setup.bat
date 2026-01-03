@echo off
echo ========================================
echo    Medical Chatbot Setup Script
echo ========================================
echo.

echo Step 1: Installing Python dependencies...
pip install -r requirements.txt
echo.

echo Step 2: Checking system requirements...
python setup_check.py
echo.

echo Step 3: Ready to build vector store!
echo Run: python vectorstore.py
echo Then: streamlit run app.py
echo.

pause
