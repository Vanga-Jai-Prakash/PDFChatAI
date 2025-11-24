**PDF Chatbot**

A simple AI app that lets you upload a PDF and ask questions about it.
The app reads the PDF, finds the most relevant text using FAISS, and uses Groq’s Llama-3 model to generate answers based only on the document content.

--> Features

Upload any PDF

Ask questions in natural language

AI answers using only the PDF text

Fast vector search using FAISS

Clean Streamlit interface

--> How it Works

Extract text from the PDF

Split it into chunks

Convert chunks into embeddings

Store them in FAISS

User asks a question

Retrieve best matching chunks

LLM (Groq) generates the answer

--> Tech Used

Streamlit

PyPDF2

SentenceTransformers

FAISS

Groq Llama-3
