# Document Assistant

An AI-powered document assistant built with Python and Flask. Upload a PDF or TXT file and ask questions about it.

## Features
- Upload PDF or TXT documents
- Ask questions about document content
- Powered by Llama 3.3 70B via Groq API
- Clean browser-based UI

## Tech Stack
- Python, Flask, Groq API, PyPDF2, HTML/CSS/JavaScript

## How to Run
1. Clone the repo
2. Add your Groq API key to a `.env` file
3. Run `pip install flask groq python-dotenv pypdf2`
4. Run `python app.py`
5. Open `http://127.0.0.1:5000`