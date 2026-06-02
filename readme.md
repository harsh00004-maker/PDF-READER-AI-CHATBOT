# 📄 AI PDF Chatbot

An AI-powered PDF Chatbot built using Streamlit, FAISS, Sentence Transformers, and Google Gemini. Users can upload a PDF document and ask questions related to its content. The chatbot retrieves relevant information from the PDF using Retrieval-Augmented Generation (RAG) and generates accurate answers.

## 🚀 Features

* Upload PDF documents
* Extract text from PDFs
* Create semantic embeddings using Sentence Transformers
* Store and search embeddings with FAISS
* Retrieve relevant PDF content based on user queries
* Generate answers using Google Gemini
* Simple and interactive Streamlit interface

## 🛠️ Tech Stack

* Python
* Streamlit
* Google Gemini API
* FAISS
* Sentence Transformers
* PyPDF2
* NumPy

## 📂 Project Workflow

1. User uploads a PDF document.
2. Text is extracted from the PDF.
3. Extracted text is divided into chunks.
4. Embeddings are generated using Sentence Transformers.
5. Embeddings are stored in a FAISS vector database.
6. User asks a question.
7. Relevant chunks are retrieved from FAISS.
8. Gemini generates an answer using the retrieved context.

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

## 📸 Use Case

This project can be used for:

* Research paper Q&A
* Resume analysis
* Report understanding
* Academic document assistance
* Knowledge retrieval from PDFs

## 👨‍💻 Author

**Harsh Singh Gaur**

Final Year Student | AI/ML Enthusiast | Data Analytics & Machine Learning
