import streamlit as st
import os
import fitz  # PyMuPDF for PDF handling
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Set up your local folder to store uploaded documents and ChromaDB
UPLOAD_FOLDER = 'uploaded_docs/'
CHROMA_DB_FOLDER = 'chroma_db/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_FOLDER, exist_ok=True)

# Define constants for chunking
CHUNK_SIZE = 500  # Max characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks

# Function to clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    text = re.sub(r'[^A-Za-z0-9\s.,?!]', '', text)  # Keep only text and punctuation
    return text.strip()

# Function to split a document into smaller chunks with overlap
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split the document text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap  # Move start with overlap
    return chunks

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

# Function to initialize ChromaDB and collection with persistence
def init_chromadb():
    # Initialize ChromaDB client with persistent storage
    client = chromadb.Client(Settings(persist_directory=CHROMA_DB_FOLDER))
    
    # Get the existing collection or create a new one
    collection = None
    try:
        # Try to retrieve the collection if it exists
        collection = client.get_collection("document_embeddings")
        st.info("Using existing 'document_embeddings' collection.")
    except Exception as e:
        # If retrieval fails, create a new collection
        st.info("Creating a new 'document_embeddings' collection.")
        collection = client.create_collection("document_embeddings")
    
    return client, collection

# Function to compute and save embeddings to ChromaDB
def compute_embeddings_and_save_to_chromadb(docs, doc_ids, collection):
    try:
        # Initialize the lightweight SentenceTransformer embedding model
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        for i, doc in enumerate(docs):
            chunks = chunk_text(doc)  # Split document into chunks
            
            # Compute embeddings for each chunk and convert to list
            embeddings = [embedder.encode(chunk).tolist() for chunk in chunks]  # Convert ndarray to list
            
            # Add chunks and embeddings to the ChromaDB collection
            for j, embedding in enumerate(embeddings):
                collection.add(
                    documents=[chunks[j]], 
                    embeddings=[embedding],  # Ensure embeddings are Python lists
                    ids=[f"{doc_ids[i]}_chunk_{j}"]  # Use chunk ID for uniqueness
                )
        
        st.success("Embeddings saved successfully to ChromaDB.")
        
        # Persistence happens at the client level
        collection.client.persist()  # Ensure persistence of the entire ChromaDB
    except Exception as e:
        st.error(f"Error saving embeddings to ChromaDB: {e}")

# Function to load the question-answer model using Hugging Face transformers (FLAN-T5 model)
def load_rag_model():
    try:
        # Use Hugging Face for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load the ChromaDB vector store for retrieving relevant documents
        vector_store = Chroma("document_embeddings", embeddings, persist_directory=CHROMA_DB_FOLDER)
        
        # Initialize a question-answering pipeline using the FLAN-T5 model
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", tokenizer="google/flan-t5-small")
        
        return qa_pipeline, vector_store
    except Exception as e:
        st.error(f"Error loading RAG model: {e}")
        return None, None

# Streamlit UI
st.title("Document Uploader & Cleaner with RAG and Chunking")

# Upload Section
uploaded_files = st.file_uploader("Upload PDFs or Text Files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Uploaded Documents")
    docs = []
    doc_ids = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Extract and clean the document text
        try:
            if uploaded_file.name.endswith('.pdf'):
                doc_text = extract_text_from_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_text = clean_text(f.read())

            docs.append(doc_text)
            doc_ids.append(uploaded_file.name)
            
            # Display uploaded document names
            st.write(f"Document: {uploaded_file.name} uploaded and cleaned.")
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

    # Vectorize and save documents to ChromaDB
    if st.button("Vectorize and Save Documents"):
        try:
            client, collection = init_chromadb()  # Initialize ChromaDB and collection
            compute_embeddings_and_save_to_chromadb(docs, doc_ids, collection)  # Compute and save embeddings
        except Exception as e:
            st.error(f"Error vectorizing and saving documents: {e}")

    # Retrieve using RAG (Retrieve-Answer-Generate)
    st.subheader("Ask Questions about the Documents")
    question = st.text_input("Enter your question:")

    if st.button("Ask"):
        try:
            qa_pipeline, vector_store = load_rag_model()
            if qa_pipeline and vector_store:
                # Perform a similarity search and retrieve the top 3 most relevant chunks
                docs_retrieved = vector_store.similarity_search(question, k=3)  # Limit to 3 docs for speed
                
                # Combine the retrieved chunks for the context
                context = ' '.join([doc.page_content for doc in docs_retrieved])
                
                # Generate the answer using the QA pipeline
                answer = qa_pipeline(question + " " + context)[0]['generated_text']
                st.write(f"Answer: {answer}")
            else:
                st.error("Error loading RAG model. Please check the error messages above.")
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
