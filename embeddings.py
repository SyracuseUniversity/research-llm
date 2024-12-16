import pandas as pd
import requests
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv('.env.local')

# Load storage path for ChromaDB
storage_path = os.getenv('STORAGE_PATH')
if storage_path is None:
    raise ValueError('STORAGE_PATH environment variable is not set')

# Initialize PersistentClient for ChromaDB
client = chromadb.PersistentClient(path=storage_path)
# Delete the collection
client.delete_collection(name="research_papers")
collection = client.get_or_create_collection(name="research_papers")

# Read CSV into DataFrame
df = pd.read_csv('syracuse_university_orcid_data.csv')

arxiv_df = df[df['work_url'].str.contains('arxiv', na=False)].copy()
arxiv_df['work_url'] = arxiv_df['work_url'].str.replace('http://arxiv.org/abs', 'http://arxiv.org/pdf')

# Function to generate a unique UUID for each document
def generate_unique_id():
    return str(uuid.uuid4())

# Function to extract text from PDF
def extract_pdf_text(url):
    response = requests.get(url)
    with open("temp.pdf", 'wb') as f:
        f.write(response.content)
    with open("temp.pdf", 'rb') as f:
        reader = PdfReader(f)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to get embeddings for text
def get_text_embeddings(text):
    embedding = model.encode([text])[0]
    return embedding



# Iterate over filtered rows and add them to ChromaDB
for _, row in arxiv_df.iterrows():
    title = row['work_title']
    full_name = row['full_name']
    url = row['work_url']

    # Extract PDF content
    pdf_text = extract_pdf_text(url)
    # Generate a unique UUID for the document
    unique_id = generate_unique_id()
    # Get embeddings for the text content
    embedding = get_text_embeddings(pdf_text)

    # Add to ChromaDB collection
    collection.add(
        documents=[pdf_text],
        metadatas=[{"title": title, "full_name": full_name}],
        ids=[url],
        embeddings=[embedding]
    )

print('Successfully added docs to collection')
# Example query: Find documents related to 'physics' in the embeddings
#query = "physics"
#query_embedding = get_text_embeddings(query)

# Query ChromaDB collection for similar embeddings
#results = collection.query(query_embeddings=[query_embedding], n_results=3)

# Print out query results
#for result in results['documents']:
 #   print(result)
  #  print('done with doc')
