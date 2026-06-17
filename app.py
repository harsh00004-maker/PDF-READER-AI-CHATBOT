import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

#  Credentials load karein
creds_dict = st.secrets["streamlit-gcp@enhanced-mote-499706-i1.iam.gserviceaccount.com"]
credentials = service_account.Credentials.from_service_account_info(creds_dict)

# Yahan project id daal di hai
client = storage.Client(credentials=credentials, project="enhanced-mote-499706-i1")

# Load environment variables
load_dotenv()
print("KEY =", os.getenv("GEMINI_API_KEY"))

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Gemini 1.5 Flash Model
model = genai.GenerativeModel(
    "models/gemini-3.5-flash"
)

# Streamlit Page Config
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄"
)

st.title("📄 AI PDF Chatbot")
st.write("Upload a PDF and ask questions from it.")

# Load Embedding Model Once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    if not text.strip():
        st.error("No text found in PDF.")
        st.stop()

    # Chunking
    chunk_size = 500
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(
            text[i:i + chunk_size]
        )

    # Create Embeddings
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # Create FAISS Index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    st.success(
        "✅ PDF processed successfully!"
    )

    # User Question
    question = st.text_input(
        "Ask a question from the PDF"
    )

    if question:

        # Question Embedding
        question_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        question_embedding = question_embedding.astype(
            "float32"
        )

        # Similarity Search
        D, I = index.search(
            question_embedding,
            k=3
        )

        # Retrieve Relevant Chunks
        relevant_chunks = []

        for idx in I[0]:
            relevant_chunks.append(
                chunks[idx]
            )

        context = "\n".join(
            relevant_chunks
        )

        # Prompt
        prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I could not find the answer in the uploaded PDF."
"""

        try:
            response = model.generate_content(
                prompt
            )

            st.subheader(
                "Answer"
            )

            st.write(
                response.text
            )

        except Exception as e:
            st.error(
                f"Error: {e}"
            )
