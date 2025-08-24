import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Get the maximum number of documents from environment variables (default: no limit)
MAX_DOCUMENTS = os.getenv("MAX_DOCUMENTS")
if MAX_DOCUMENTS is not None:
    MAX_DOCUMENTS = int(MAX_DOCUMENTS)

def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "tujuhsembilan"),
            user=os.getenv("DB_USER", "admin"),
            password=os.getenv("DB_PASS", "password")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def init_db():
    """Initialize the database tables."""
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            # Create documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding JSONB,  -- Store embedding as JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        conn.close()
        return False

def add_document(content, metadata=None, embedding=None):
    """Add a document to the database."""
    # Check if we've reached the maximum document limit (if set)
    if MAX_DOCUMENTS is not None and get_document_count() >= MAX_DOCUMENTS:
        print(f"Document limit reached ({MAX_DOCUMENTS} documents). Cannot add more documents.")
        return None
    
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (content, json.dumps(metadata) if metadata else None, 
                  json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding) if embedding is not None else None))
            
            doc_id = cur.fetchone()[0]
            conn.commit()
        conn.close()
        return doc_id
    except Exception as e:
        print(f"Error adding document: {e}")
        conn.close()
        return None

def get_all_documents():
    """Retrieve all documents from the database."""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents ORDER BY id")
            documents = cur.fetchall()
        conn.close()
        
        # Convert embedding JSON back to numpy array
        for doc in documents:
            if doc['embedding'] is not None:
                doc['embedding'] = np.array(doc['embedding'])
                
        return documents
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        conn.close()
        return []

def get_document_by_id(doc_id):
    """Retrieve a document by its ID from the database."""
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
            document = cur.fetchone()
        conn.close()
        
        # Convert embedding JSON back to numpy array
        if document and document['embedding'] is not None:
            document['embedding'] = np.array(document['embedding'])
                
        return document
    except Exception as e:
        print(f"Error retrieving document: {e}")
        conn.close()
        return None

def get_document_count():
    """Get the count of documents in the database."""
    conn = get_db_connection()
    if not conn:
        return 0
        
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            count = cur.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting document count: {e}")
        conn.close()
        return 0