# Incremental RAG with Qwen Models

This is an incremental Retrieval-Augmented Generation (RAG) system using Qwen models via Ollama with PostgreSQL storage. The system allows you to:

1. Add documents incrementally to build your knowledge base
2. Query the system to find relevant documents
3. Chat with the model using the stored documents as context

## Features

- **Incremental Updates**: Add documents to your knowledge base at any time
- **Vector Search**: Uses FAISS for efficient similarity search
- **Web Interface**: Simple web UI for interacting with the system
- **API Access**: RESTful API for programmatic access
- **Persistent Storage**: Documents and embeddings stored in PostgreSQL database

## Prerequisites

1. Python 3.7 or higher
2. Ollama already running (in Docker or locally)
3. PostgreSQL database already running (in Docker or locally)
4. Qwen models pulled in Ollama:
   ```bash
   ollama pull qwen3:latest
   ollama pull qwen3:4b-instruct
   ```

## Installation

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

1. Run the web API:
   ```bash
   python api.py
   ```

2. Open your browser and go to http://localhost:8000 to access the web interface.

You can:
- Add documents to the knowledge base
- Chat with the model using RAG context
- View all stored documents

### Simple Chatbot

Run the simple chatbot (no document/RAG capabilities):
```bash
python chatbot.py
```

This provides a basic chat interface with the Qwen model without any document context.

### Advanced Chatbot

Run the advanced chatbot (with optional RAG capabilities):
```bash
python advanced_chatbot.py
```

This provides a chat interface that can optionally use documents for context when available.

#### Commands

- `add <content>` - Add content to the knowledge base
- `rag on/off` - Enable/disable RAG mode
- `count` - Show document count
- `list` - List all documents
- `clear` - Clear the screen
- `help` - Show available commands
- `quit` - Exit the program

### Command-Line Interface

Run the RAG application:
```bash
python rag_app_db.py
```

#### Commands

- `add <content>` - Add a document to the RAG
- `chat <query>` - Chat with the model (with RAG context)
- `direct <query>` - Chat with the model (without RAG context)
- `search <query>` - Search for similar documents
- `count` - Show document count
- `list` - List all documents
- `help` - Show this help
- `quit` - Exit the program

#### Examples

```
> add The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
Added document with ID: 1

> add The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor.
Added document with ID: 2

> chat What is the Eiffel Tower?
RAG Response: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.

> search statue
Similar documents:
  1. (ID: 2, Score: 0.8765) The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor....
```

### API Endpoints

When running the API server (`python api.py`), the following endpoints are available:

- `POST /add` - Add a document
  ```json
  {
    "content": "Document content here",
    "metadata": {"source": "user"}
  }
  ```

- `POST /chat` - Chat with RAG context
  ```json
  {
    "query": "Your question here"
  }
  ```

- `POST /direct` - Chat without RAG context
  ```json
  {
    "query": "Your question here"
  }
  ```

- `POST /search` - Search for similar documents
  ```json
  {
    "query": "Search query",
    "k": 5
  }
  ```

- `GET /count` - Get document count

- `GET /documents` - List all documents

### Testing Document Limit

To test the document limit functionality:

```bash
python test_limit.py
```

This script will attempt to add documents until the limit is reached.

To enable document limiting for testing:
1. Uncomment `MAX_DOCUMENTS=10` in the `.env` file
2. Run the test script

For production use, ensure `MAX_DOCUMENTS` is commented out or removed.

## Configuration

The application reads configuration from `.env` file:
- Database configuration:
  - `DB_HOST`: PostgreSQL host (default: localhost)
  - `DB_PORT`: PostgreSQL port (default: 5432)
  - `DB_NAME`: PostgreSQL database name (default: tujuhsembilan)
  - `DB_USER`: PostgreSQL user (default: admin)
  - `DB_PASS`: PostgreSQL password (default: password)
- Ollama configuration:
  - `MODEL_EMB`: Model for embeddings (default: qwen3:4b-instruct)
  - `MODEL_CHAT`: Model for chat (default: qwen3:latest)
  - `OLLAMA_HOST`: Ollama service host (default: http://localhost:11434)
  - `OLLAMA_TIMEOUT`: Timeout for Ollama requests (default: 60)
  - `OLLAMA_KEEP_ALIVE`: Keep alive time for Ollama models (default: 10m)
- Testing configuration:
  - `MAX_DOCUMENTS`: Maximum number of documents allowed (default: unlimited)

### Testing vs Production Mode

- For **testing mode** (limited to 10 documents):
  ```
  MAX_DOCUMENTS=10
  ```

- For **production mode** (unlimited documents):
  ```
  # MAX_DOCUMENTS=10
  ```
  or remove the line entirely.

## Project Structure

```
.
├── .env                 # Configuration file
├── requirements.txt     # Python dependencies
├── rag_app_db.py        # Main RAG application (CLI) with PostgreSQL
├── api.py               # Web API
├── db.py                # Database connection and operations
├── index.html           # Web interface
└── README.md            # This file
```