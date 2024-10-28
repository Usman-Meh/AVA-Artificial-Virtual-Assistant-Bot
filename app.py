import faiss
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Initialize Flask app
app = Flask(__name__)

# Path to the 'static' folder (where your CSV and FAISS files are stored)
STATIC_DIR = os.path.join(app.root_path, 'static')

# Load the RAG data from CSV stored in the static folder
rag_data_file = os.path.join(STATIC_DIR, "chunks.csv")
rag_data = pd.read_csv(rag_data_file, header=None, names=["text"])  # Assuming there's no header row in CSV


def load_pdf():
    # Path to the PDF file in the static folder
    pdf_path = os.path.join('static', 'AI_file.pdf')

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    
    # Extract text from the PDF pages
    pages = loader.load()
    chunk_size=300 # chunk size is for making limit of character  single chunk
    chunk_overlap=10  # reading previous character of chunk to connect information
   
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split the text into manageable chunks (this will return Document objects)
    document_chunks = text_splitter.split_documents(pages)

    # Extract only the 'page_content' (text) from each Document object
    chunks = [doc.page_content for doc in document_chunks]
    
    return chunks


def load_model(query):
    # Get the absolute path to the saved_model folder in the static directory
    model_path = os.path.join(app.root_path, 'static', 'saved_model')

    
    # Load the model
    model = SentenceTransformer(model_path)
    query_embedding = model.encode([query])
    return query_embedding

def retrieve_answer(query: str, rag_data, k=10):
    """Retrieve top-k similar answers from FAISS index based on query embedding."""
        
    # Load the FAISS index from the static folder
    index = os.path.join(STATIC_DIR, "saved_faiss.index")

    try:
        index = faiss.read_index(index)
        print(f"FAISS index type: {type(index)}")
        
        # Check if the 'd' attribute exists
        if hasattr(index, 'd'):
            print(f"FAISS index loaded, dimensionality: {index.d}")
        else:
            print("The FAISS index does not have a 'd' attribute.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        
        
    query_embedding = load_model(query)
    
    # Search for top-k nearest neighbors in FAISS index
    distances, indices = index.search(np.array(query_embedding), k)
    #print("Distance:",distances, "  Indices: ",indices)
    chunks =load_pdf()
    # Retrieve corresponding text data from CSV based on index positions
    relevant_chunks = [chunks[i] for i in indices[0]]

    # Combine the chunks into a context for generation
    relevant_chunks = " ".join(relevant_chunks)
    
    return relevant_chunks if relevant_chunks else ["No relevant answer found."]


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/retrieve', methods=['POST'])
def retrieve():
    query = request.form.get("query")
    if query:
        try:
            response = retrieve_answer(query, rag_data, k=5)
            
            # Split response into words for timed output
            response_words = response.split()
            
            # Return response to be displayed word-by-word in the HTML
            return render_template('index.html', responses=response_words)
        except Exception as e:
            return render_template('index.html', responses=[f"Error: {e}"])
    return render_template('index.html', responses=["No query provided."])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)
