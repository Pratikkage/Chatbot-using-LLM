import os
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import weaviate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Download necessary NLTK data
nltk.download('stopwords')

# -------------------------------
# Step 1: Web Scraping and Cleaning
# -------------------------------

def scrape_website(url):
    """Scrape text content from a given website."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text
    else:
        raise Exception(f"Failed to fetch URL: {url}, Status Code: {response.status_code}")

def clean_text(text):
    """Clean and preprocess raw text."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# -------------------------------
# Step 2: Embedding Generation
# -------------------------------

def generate_embeddings(text):
    """Generate embeddings using a Sentence Transformer model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text, convert_to_numpy=True)
    return embeddings

# -------------------------------
# Step 3: Initialize Vector Database
# -------------------------------

def initialize_weaviate():
    """Initialize a connection to the Weaviate vector database."""
    client = weaviate.Client(url="http://localhost:8080")
    return client

def create_schema(client):
    """Create schema for storing documents in Weaviate."""
    schema = {
        "classes": [
            {
                "class": "Document",
                "properties": [
                    {"name": "text", "dataType": ["text"]},
                    {"name": "embedding", "dataType": ["number[]"]}
                ]
            }
        ]
    }
    client.schema.create(schema)

def store_embeddings(client, embeddings_data):
    """Store document text and embeddings in Weaviate."""
    for text, embedding in embeddings_data:
        client.data_object.create({"text": text, "embedding": embedding.tolist()}, "Document")

# -------------------------------
# Step 4: Initialize Retrieval-Augmented QA System
# -------------------------------

def initialize_qa_system(api_key, retriever):
    """Initialize a LangChain QA system with an OpenAI LLM."""
    llm = OpenAI(api_key=api_key)
    return RetrievalQA(llm=llm, retriever=retriever)

def chatbot_query(qa_system, query):
    """Handle user queries through the QA system."""
    response = qa_system(query)
    return response['answer']

# -------------------------------
# Step 5: Deploying the Chatbot via Flask
# -------------------------------

app = Flask(__name__)

# Configuration: URLs for scraping
urls = [
    "https://www.changiairport.com",
    "https://www.jewelchangiairport.com"
]

# Initialize Vector Database and create schema
try:
    client = initialize_weaviate()
    create_schema(client)
    print("Weaviate client initialized and schema created.")
except Exception as e:
    print(f"Error initializing Weaviate: {e}")

# Scrape and preprocess website data, then store embeddings
try:
    embeddings_data = []
    for url in urls:
        raw_text = scrape_website(url)
        clean_data = clean_text(raw_text)
        embedding = generate_embeddings(clean_data)
        embeddings_data.append((clean_data, embedding))
    store_embeddings(client, embeddings_data)
    print("Data scraped, processed, and stored in Weaviate.")
except Exception as e:
    print(f"Error during data processing or storage: {e}")

# Initialize QA system
try:
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure the OpenAI API key is set as an environment variable
    qa_system = initialize_qa_system(api_key=api_key, retriever=client)
    print("QA system initialized.")
except Exception as e:
    print(f"Error initializing QA system: {e}")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """REST API endpoint for chatbot interaction."""
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'Query not provided'}), 400
    try:
        answer = chatbot_query(qa_system, query)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)