import os
import json
import numpy as np
import faiss
import ollama
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import time
from datetime import datetime
from db import get_db_connection, init_db, add_document, get_all_documents, get_document_count, get_document_by_id

# Load environment variables
load_dotenv()

# Get RAG configuration from environment variables
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 10))
RAG_CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE", 1000))
RAG_DEFAULT_DIMENSION = int(os.getenv("RAG_DEFAULT_DIMENSION", 128))


class IncrementalRAG:
    def __init__(self, embedding_model: str = None, chat_model: str = None, ollama_host: str = None):
        """
        Initialize the Incremental RAG system with PostgreSQL storage.
        
        Args:
            embedding_model: Model name for generating embeddings
            chat_model: Model name for chat responses
            ollama_host: Ollama host URL (default: http://localhost:11434)
        """
        
        # Load model names from environment variables or use defaults
        self.embedding_model = embedding_model or os.getenv("MODEL_EMB", "qwen3:4b-instruct")
        self.chat_model = chat_model or os.getenv("MODEL_CHAT", "qwen3:latest")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        print(" IncrementalRAG init using {self.embedding_model}") 
        # Initialize FAISS index for vector storage
        # Using L2 distance for similarity search
        self.dimension = RAG_DEFAULT_DIMENSION  # Default dimension, will be updated after first embedding
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize embedding cache
        self.embedding_cache = {}
        
        # Initialize database
        init_db()
        
        # Load existing documents and embeddings from database
        self._load_from_db()
    
    def _load_from_db(self):
        """Load documents and embeddings from the database."""
        documents = get_all_documents()
        
        if not documents:
            return
            
        # Update dimension based on first embedding
        first_doc = documents[0]
        if first_doc['embedding'] is not None:
            self.dimension = len(first_doc['embedding'])
            # Reinitialize index with correct dimension if needed
            if self.index.d != self.dimension:
                self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add all embeddings to the FAISS index
        embeddings = []
        for doc in documents:
            if doc['embedding'] is not None:
                embeddings.append(doc['embedding'])
                
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.index.add(embeddings_array)
            
        print(f"Loaded {len(documents)} documents from database")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using the specified model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        # Check if embedding is in cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            # Set the host for ollama client
            import ollama
            client = ollama.Client(host=self.ollama_host)
            response = client.embeddings(model=self.embedding_model, prompt=text)
            embedding = np.array(response["embedding"], dtype=np.float32)
            print(" _get_embedding using {self.embedding_model}")
            # Update dimension if this is the first embedding
            if self.index.ntotal == 0 and self.dimension != len(embedding):
                self.dimension = len(embedding)
                # Reinitialize index with correct dimension
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache the embedding, but limit cache size to RAG_CACHE_SIZE from .env
            if len(self.embedding_cache) > RAG_CACHE_SIZE:
                # Remove the first item (oldest) from the cache
                first_key = next(iter(self.embedding_cache))
                del self.embedding_cache[first_key]
            
            self.embedding_cache[text] = embedding
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector if embedding fails
            return np.zeros(self.dimension, dtype=np.float32)
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> int:
        """
        Add a document to the RAG system.
        
        Args:
            content: Document content
            metadata: Additional metadata for the document
            
        Returns:
            Document ID
        """
        # Generate embedding for the document
        embedding = self._get_embedding(content)
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Add to database
        doc_id = add_document(content, metadata, embedding)
        
        return doc_id
    
    def search_similar(self, query: str, k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of similar documents to return (defaults to RAG_TOP_K from .env)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Use the provided k value or default to RAG_TOP_K from .env
        if k is None:
            k = RAG_TOP_K
            
        # Get embedding for the query
        query_embedding = self._get_embedding(query)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(k, self.index.ntotal))
        
        # Retrieve documents from database - only the ones we need
        from db import get_document_by_id, get_all_documents
        documents = []
        
        # Get all documents to map index to ID
        all_docs = get_all_documents()
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(all_docs):  # Check bounds
                doc = all_docs[idx]
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1 / (1 + distance)
                results.append((dict(doc), similarity))
        
        return results
    
    def chat(self, query: str, use_rag: bool = True) -> str:
        """
        Chat with the model, optionally using RAG context.
        
        Args:
            query: User query
            use_rag: Whether to use RAG context
            
        Returns:
            Model response
        """
        # If using RAG and we have documents, retrieve relevant context
        context = ""
        if use_rag:
            # Print message when sending to RAG
            print(f"Send TO RAG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            similar_docs = self.search_similar(query)
            if similar_docs:
                context = "\n\nRelevant context:\n"
                for doc, similarity in similar_docs:
                    context += f"- {doc['content']}\n"
        
        # Prepare the prompt
        if context:
            prompt = f"Answer the query using the provided context.\n\nQuery: {query}\n{context}\n\nAnswer:"
        else:
            prompt = query
        
        try:
            # Generate response using the chat model
            import ollama
            client = ollama.Client(host=self.ollama_host)
            response = client.generate(model=self.chat_model, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            return f"Error generating response: {e}"
    
    def get_document_count(self) -> int:
        """Get the number of documents in the RAG system."""
        return get_document_count()
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the RAG system."""
        return get_all_documents()


def main():
    """Main function to demonstrate the RAG system."""
    # Initialize RAG system
    rag = IncrementalRAG()
    
    print("=== Incremental RAG System with Qwen Models ===")
    print("Commands:")
    print("  add <content>     - Add a document to the RAG")
    print("  chat <query>      - Chat with the model (with RAG)")
    print("  direct <query>    - Chat with the model (without RAG)")
    print("  search <query>    - Search for similar documents")
    print("  count             - Show document count")
    print("  list              - List all documents")
    print("  help              - Show this help")
    print("  quit              - Exit the program")
    print()
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "quit":
                break
                
            if user_input.lower() == "help":
                print("Commands:")
                print("  add <content>     - Add a document to the RAG")
                print("  chat <query>      - Chat with the model (with RAG)")
                print("  direct <query>    - Chat with the model (without RAG)")
                print("  search <query>    - Search for similar documents")
                print("  count             - Show document count")
                print("  list              - List all documents")
                print("  help              - Show this help")
                print("  quit              - Exit the program")
                continue
            
            if user_input.lower().startswith("add "):
                content = user_input[4:].strip()
                if content:
                    start_time = time.time()
                    doc_id = rag.add_document(content)
                    end_time = time.time()
                    response_time = end_time - start_time
                    print(f"Added document with ID: {doc_id}")
                    print(f"[Response time: {response_time:.2f} seconds]")
                else:
                    print("Please provide content to add")
                continue
                
            if user_input.lower().startswith("chat "):
                query = user_input[5:].strip()
                if query:
                    start_time = time.time()
                    response = rag.chat(query, use_rag=True)
                    end_time = time.time()
                    response_time = end_time - start_time
                    print(f"RAG Response: {response}")
                    print(f"[Response time: {response_time:.2f} seconds]")
                else:
                    print("Please provide a query")
                continue
                
            if user_input.lower().startswith("direct "):
                query = user_input[7:].strip()
                if query:
                    start_time = time.time()
                    response = rag.chat(query, use_rag=False)
                    end_time = time.time()
                    response_time = end_time - start_time
                    print(f"Direct Response: {response}")
                    print(f"[Response time: {response_time:.2f} seconds]")
                else:
                    print("Please provide a query")
                continue
                
            if user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                if query:
                    start_time = time.time()
                    results = rag.search_similar(query, k=3)
                    end_time = time.time()
                    response_time = end_time - start_time
                    if results:
                        print("Similar documents:")
                        for i, (doc, similarity) in enumerate(results, 1):
                            print(f"  {i}. (ID: {doc['id']}, Score: {similarity:.4f}) {doc['content'][:100]}...")
                    else:
                        print("No similar documents found")
                    print(f"[Response time: {response_time:.2f} seconds]")
                else:
                    print("Please provide a query")
                continue
                
            if user_input.lower() == "count":
                start_time = time.time()
                count = rag.get_document_count()
                end_time = time.time()
                response_time = end_time - start_time
                print(f"Document count: {count}")
                print(f"[Response time: {response_time:.2f} seconds]")
                continue
                
            if user_input.lower() == "list":
                start_time = time.time()
                docs = rag.list_documents()
                end_time = time.time()
                response_time = end_time - start_time
                if docs:
                    print("Documents:")
                    for doc in docs:
                        print(f"  ID: {doc['id']}, Content: {doc['content'][:100]}...")
                else:
                    print("No documents")
                print(f"[Response time: {response_time:.2f} seconds]")
                continue
                
            print("Unknown command. Type 'help' for available commands.")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()