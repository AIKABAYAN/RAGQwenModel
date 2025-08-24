import os
import time
from datetime import datetime
from flask import Flask, request, jsonify
from rag_app_db import IncrementalRAG

app = Flask(__name__)

# Initialize RAG system with Ollama host from environment
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
rag = IncrementalRAG(ollama_host=ollama_host)

# Get the maximum number of documents from environment variables (default: no limit)
MAX_DOCUMENTS = os.getenv("MAX_DOCUMENTS")
if MAX_DOCUMENTS is not None:
    MAX_DOCUMENTS = int(MAX_DOCUMENTS)

@app.route('/add', methods=['POST'])
def add_document():
    """Add a document to the RAG system."""
    data = request.get_json()
    content = data.get('content', '')
    metadata = data.get('metadata', {})
    
    if not content:
        return jsonify({'error': 'Content is required'}), 400
    
    # Check if we've reached the maximum document limit (if set)
    if MAX_DOCUMENTS is not None and rag.get_document_count() >= MAX_DOCUMENTS:
        return jsonify({'error': f'Document limit reached ({MAX_DOCUMENTS} documents). Cannot add more documents.'}), 400
    
    start_time = time.time()
    doc_id = rag.add_document(content, metadata)
    end_time = time.time()
    response_time = end_time - start_time
    
    return jsonify({'id': doc_id, 'message': 'Document added successfully', 'response_time': response_time})

@app.route('/chat', methods=['POST'])
def chat():
    """Chat with the model using RAG context."""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Log that we're sending to RAG
    print(f"Send TO RAG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    response = rag.chat(query, use_rag=True)
    end_time = time.time()
    response_time = end_time - start_time
    
    return jsonify({'response': response, 'response_time': response_time})

@app.route('/direct', methods=['POST'])
def direct_chat():
    """Chat with the model without RAG context."""
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    start_time = time.time()
    response = rag.chat(query, use_rag=False)
    end_time = time.time()
    response_time = end_time - start_time
    
    return jsonify({'response': response, 'response_time': response_time})

@app.route('/search', methods=['POST'])
def search():
    """Search for similar documents."""
    data = request.get_json()
    query = data.get('query', '')
    k = data.get('k', 10)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Log that we're sending to RAG
    print(f"Send TO RAG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    results = rag.search_similar(query, k)
    end_time = time.time()
    response_time = end_time - start_time
    
    formatted_results = [
        {
            'document': doc,
            'similarity': float(similarity)
        }
        for doc, similarity in results
    ]
    return jsonify({'results': formatted_results, 'response_time': response_time})

@app.route('/count', methods=['GET'])
def count():
    """Get the number of documents in the RAG system."""
    count = rag.get_document_count()
    response = {'count': count}
    if MAX_DOCUMENTS is not None:
        response['max_documents'] = MAX_DOCUMENTS
    return jsonify(response)

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in the RAG system."""
    docs = rag.list_documents()
    response = {'documents': docs}
    if MAX_DOCUMENTS is not None:
        response['max_documents'] = MAX_DOCUMENTS
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)