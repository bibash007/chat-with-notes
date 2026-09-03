# Scratch-Built RAG (Retrieval-Augmented Generation)

This project is a completely scratch-built RAG application designed to understand the fundamental architecture of how AI applications chat with private data. It does not rely on heavy frameworks like LangChain or LlamaIndex.

## Architecture

1. **Ingestion (`1_ingest.py`)**: Reads `.txt` files from the `notes/` directory, chunks them, generates vector embeddings using a local HuggingFace model (`all-MiniLM-L6-v2`), and stores them in a local ChromaDB vector database.
2. **Retrieval & Generation (`2_chat.py`)**: Takes a user query, embeds it, searches ChromaDB for the 3 most relevant chunks, and passes them as context to the Gemini API to generate an accurate response.

## Tech Stack
* **Vector Database:** ChromaDB
* **Embeddings:** `sentence-transformers` (HuggingFace)
* **LLM:** Google Gemini API
* **Language:** Python

## How to Run Locally

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
4. Add some `.txt` files with your notes to the `notes/` directory.
5. Run the ingestion script to build the database:
   ```bash
   python 1_ingest.py
   ```
6. Start chatting with your notes:
   ```bash
   python 2_chat.py
   ```
