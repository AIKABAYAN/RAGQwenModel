#!/usr/bin/env python3
"""
Simple Chatbot Interface
This script provides a simple chat interface without document limits.
"""

import os
import time
from datetime import datetime
from rag_app_db import IncrementalRAG

def main():
    print("=== Simple Chatbot ===")
    print("Type 'quit' to exit")
    print("Type 'help' for available commands")
    print()
    
    # Initialize RAG system
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    rag = IncrementalRAG(ollama_host=ollama_host)
    
    # Show document count
    count = rag.get_document_count()
    print(f"Documents in knowledge base: {count}")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
                
            if user_input.lower() == "help":
                print("Commands:")
                print("  help     - Show this help")
                print("  quit     - Exit the program")
                print("  count    - Show document count")
                print("  list     - List all documents")
                print("  clear    - Clear the screen")
                print()
                continue
                
            if user_input.lower() == "count":
                count = rag.get_document_count()
                print(f"Documents in knowledge base: {count}")
                continue
                
            if user_input.lower() == "list":
                docs = rag.list_documents()
                if docs:
                    print("Documents in knowledge base:")
                    for doc in docs:
                        print(f"  ID: {doc['id']}, Content: {doc['content'][:100]}...")
                else:
                    print("No documents in knowledge base")
                continue
                
            if user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            # Chat with the model (without RAG context for simple chatbot)
            # Note: For simple chatbot, we're not using RAG, so no "Send TO RAG" message
            start_time = time.time()
            response = rag.chat(user_input, use_rag=False)
            end_time = time.time()
            response_time = end_time - start_time
            print(f"Bot: {response}")
            print(f"[Response time: {response_time:.2f} seconds]")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()