# PDF Chatbot using LangChain & Hugging Face

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF documents and ask questions about their contents.

## Features

- Upload multiple PDFs
- Extract text from PDFs
- Automatic text chunking
- Semantic search using FAISS
- Hugging Face sentence embeddings
- Streamlit web interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face
- FAISS
- PyPDF2

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```