#!/usr/bin/env python3
"""
Test script to demonstrate the document limit functionality.
This script will add documents until the limit is reached.
"""

import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# Get test configuration from environment variables
TEST_MAX_DOCS_TO_ADD = int(os.getenv("TEST_MAX_DOCS_TO_ADD", 15))
TEST_REQUEST_DELAY = int(os.getenv("TEST_REQUEST_DELAY", 1))

def add_document(content):
    """Add a document to the RAG system."""
    response = requests.post(f"{API_BASE}/add", json={"content": content})
    return response.json(), response.status_code

def get_document_count():
    """Get the current document count."""
    response = requests.get(f"{API_BASE}/count")
    return response.json(), response.status_code

def main():
    print("=== Testing Document Limit ===")
    
    # Get initial count
    count_data, status = get_document_count()
    print(f"Initial document count: {count_data.get('count', 0)}")
    if 'max_documents' in count_data:
        print(f"Maximum documents allowed: {count_data['max_documents']}")
    else:
        print("No document limit (production mode)")
    
    # Add documents until we hit the limit or a reasonable number
    max_docs_to_add = TEST_MAX_DOCS_TO_ADD  # In production, this will add all 15 documents
    if 'max_documents' in count_data and count_data['max_documents'] is not None:
        max_docs_to_add = min(max_docs_to_add, count_data['max_documents'] + 5)
        
    for i in range(1, max_docs_to_add + 1):  # Try to add documents
        content = f"This is test document number {i} with some unique content to distinguish it from others."
        print(f"\nAdding document {i}...")
        
        result, status = add_document(content)
        
        if status == 200:
            print(f"Success: {result}")
        else:
            print(f"Error ({status}): {result}")
            # If we hit the limit, stop adding documents
            if "limit reached" in str(result):
                break
            
        # Get current count
        count_data, _ = get_document_count()
        print(f"Current document count: {count_data.get('count', 0)}")
        
        # Wait a bit between requests
        time.sleep(TEST_REQUEST_DELAY)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()