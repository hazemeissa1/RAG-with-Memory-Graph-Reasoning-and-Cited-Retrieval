```markdown
# Memory-Based, Graph-Based RAG System

A proof-of-concept for a Retrieval-Augmented Generation (RAG) system using Ollama, ChromaDB, NetworkX, and Streamlit. The system processes documents, stores them in a vector database, builds a knowledge graph, and answers queries with context from memory and graph data.

## Requirements
- Python 3.9+
- Ollama (with Llama 3 and mxbai-embed-large models)
- Dependencies listed in `requirements.txt`

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Ollama is running locally with the required models.
3. Run the Streamlit app:
   ```bash
   streamlit run app/main.py
   ```
![Description](https://github.com/user-attachments/assets/f0b84e3d-840e-483b-b01c-2dd64b282890)

## Features
- Upload and process documents (PDF, DOCX, TXT, CSV, XLSX).
- Store document chunks in ChromaDB.
- Build and query a knowledge graph using NetworkX.
- Interactive chat interface with Streamlit.
- Visualize knowledge graph for query responses.

## File Structure
- `app/`: Streamlit application logic.
- `core/`: Core RAG system components (RAG system, document processing, memory, graph, utilities).
- `config/`: Configuration files (prompt templates).
- `requirements.txt`: Project dependencies.
- `README.md`: This file.

## Usage
1. Open the Streamlit app in your browser.
2. Configure the LLM and embedding models in the sidebar.
3. Upload a document and provide a name.
4. Ask questions in the chat interface to get responses with reasoning and source citations.
5. Expand responses to view the knowledge graph and detailed reasoning.

## Author
Hazem Eissa


```
