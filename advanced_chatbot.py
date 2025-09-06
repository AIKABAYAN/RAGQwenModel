#!/usr/bin/env python3
"""
Advanced Chatbot Interface with RAG capabilities
This script provides a chat interface that can use documents when available.
"""

import os
import time
from datetime import datetime
from rag_app_db import IncrementalRAG

def main():
    print("=== Advanced Chatbot with RAG ===")
    print("Type 'quit' to exit")
    print("Type 'help' for available commands")
    print("Chat memory is enabled - previous conversations provide context")
    print()
    
    # Initialize RAG system
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    rag = IncrementalRAG(ollama_host=ollama_host)
    
    # Show document count
    count = rag.get_document_count()
    if count > 0:
        print(f"Documents in knowledge base: {count}")
        use_rag = True
        print("RAG mode: Enabled (will use documents for context)")
    else:
        use_rag = False
        print("RAG mode: Disabled (no documents available)")
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
                print("  help        - Show this help")
                print("  quit        - Exit the program")
                print("  count       - Show document count")
                print("  list        - List all documents")
                print("  clear       - Clear the screen")
                print("  rag on/off  - Enable/disable RAG mode")
                print("  add <text>  - Add text to knowledge base")
                print("  ask <query> - Ask a question using RAG context")
                print("  history     - Show chat history")
                print("  forget      - Clear chat history")
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
                
            if user_input.lower() == "history":
                history_context = rag.get_chat_history_context()
                if history_context:
                    print("Chat History:")
                    print(history_context)
                else:
                    print("No chat history available.")
                continue
                
            if user_input.lower() == "forget":
                rag.clear_chat_history()
                print("Chat history cleared.")
                continue
                
            if user_input.lower().startswith("rag "):
                if user_input[4:].lower() == "on":
                    use_rag = True
                    print("RAG mode: Enabled")
                elif user_input[4:].lower() == "off":
                    use_rag = False
                    print("RAG mode: Disabled")
                else:
                    print("Usage: rag on/off")
                continue
                
            if user_input.lower().startswith("add "):
                content = user_input[4:].strip()
                if content:
                    doc_id = rag.add_document(content)
                    if doc_id:
                        print(f"Added document with ID: {doc_id}")
                        # Enable RAG mode automatically when first document is added
                        if not use_rag:
                            use_rag = True
                            print("RAG mode: Enabled (automatically)")
                    else:
                        print("Failed to add document")
                else:
                    print("Please provide content to add")
                continue
                
            if user_input.lower().startswith("ask "):
                query = user_input[4:].strip()
                if query:
                    # Always use RAG for the 'ask' command
                    print(f"Send TO RAG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    start_time = time.time()
                    response = rag.chat(query, use_rag=True)
                    end_time = time.time()
                    response_time = end_time - start_time
                    print(f"Bot (with RAG): {response}")
                    print(f"[Response time: {response_time:.2f} seconds]")
                else:
                    print("Please provide a query")
                continue
                
            # Chat with the model
            if use_rag:
                print(f"Send TO RAG on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            start_time = time.time()
            response = rag.chat(user_input, use_rag=use_rag)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Add to chat history
            rag.add_to_chat_history(user_input, response)
            
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
