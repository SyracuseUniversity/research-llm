import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_util

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")  # Extracts text from each page
    return text

# Step 2: Generate embeddings using Sentence Transformers
def generate_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text])
    return embeddings[0]

# Step 3: Store embeddings in ChromaDB
def store_in_chromadb(embedding, document_id, metadata):
    client = chromadb.Client()
    # Create a collection (or get if exists)
    collection = client.get_or_create_collection("pdf_embeddings")

    # Add the embedding to the collection with associated metadata
    collection.add(
        documents=[document_id],
        embeddings=[embedding.tolist()],
        metadatas=[metadata]
    )
    print("Document stored in ChromaDB with ID:", document_id)

# Main script
pdf_path = "C:\Users\sonar\Downloads\IST645F24R2RPatchala.pdf"
document_id = "doc_1"

metadata = {
    "title": "deeptime: an R package that facilitates highly customizable visualizations of 1data over geologicaltime interval",
    "author": "William Gearty",
    "year": "2024",
}

# Extract text from PDF
text = extract_text_from_pdf(pdf_path)

# Generate embedding for the text
embedding = generate_embeddings(text)

# Store the embedding in ChromaDB
store_in_chromadb(embedding, document_id, metadata)
