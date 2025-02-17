import faiss
import numpy as np
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
import tempfile
import os

# Set up the Groq client with your API key
client = Groq(api_key="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")

# Initialize sentence transformer model for embedding
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the FAISS index
index = None
chunks = []

# Streamlit app header and sidebar UI elements
st.title("📄 RAG Chatbot using LLaMA-3 and FAISS")
st.sidebar.header("🔍 Upload Your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Function to process PDF and extract text
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
    return extracted_text.strip()

# Chunk the extracted text into smaller pieces
def split_into_chunks(text, max_chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]

# If a file is uploaded, process the PDF
if uploaded_file:
    # Create a temporary file for the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        temp_pdf_path = tmp_pdf.name

    # Extract text from the uploaded PDF
    document_text = extract_text(temp_pdf_path)
    os.remove(temp_pdf_path)  # Clean up temporary PDF

    if not document_text:
        st.error("⚠ Text extraction failed. Please upload a different PDF or use OCR.")
    else:
        st.success("✅ Text extraction successful!")
        # Split the text into chunks for processing
        chunks = split_into_chunks(document_text)

        if chunks:
            # Generate embeddings for the chunks of text
            embeddings = embed_model.encode(chunks)
            embeddings = np.array(embeddings)

            # Create the FAISS index and add the embeddings
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            st.success("✅ FAISS index has been created successfully!")

# Function to fetch the top-k relevant chunks for a query
def get_relevant_chunks(query, top_k=3):
    if index is None:
        return ["FAISS index not available. Please upload a PDF."]
    
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[idx] for idx in indices[0]]

# Function to generate a response based on the query and relevant context
def get_chatbot_response(query):
    relevant_chunks = get_relevant_chunks(query)
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"Context:\n{context}\n\nUser Query: {query}\n\nResponse:"
    
    # Use Groq API to generate a response from LLaMA-3
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "You are an AI assistant providing helpful responses."},
                  {"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True
    )

    response_text = ""
    for chunk in completion:
        response_text += chunk.choices[0].delta.content or ""
    
    return response_text

# UI for user query
if uploaded_file:
    user_query = st.text_input("💬 Ask a question about the uploaded document:")

    if user_query:
        with st.spinner("⏳ Generating response..."):
            answer = get_chatbot_response(user_query)
            st.subheader("🤖 Chatbot Response:")
            st.write(answer)