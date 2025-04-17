from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from datetime import datetime

def get_or_create_collection(chroma_client: chromadb.Client, collection_name: str, embedding_model: str) -> Any:
    """
    Get or create a ChromaDB collection.

    Args:
        chroma_client: ChromaDB client
        collection_name: Name of the collection
        embedding_model: Name of the embedding model

    Returns:
        ChromaDB collection
    """
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.OllamaEmbeddingFunction(
                model_name=embedding_model,
                url="http://localhost:11434"
            )
        )
    except:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_functions.OllamaEmbeddingFunction(
                model_name=embedding_model,
                url="http://localhost:11434"
            )
        )
    return collection

def add_to_memory(chroma_client: chromadb.Client, collection_name: str, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embedding_model: str) -> None:
    """
    Add texts to the memory store (ChromaDB).

    Args:
        chroma_client: ChromaDB client
        collection_name: ChromaDB collection name
        texts: List of text chunks
        metadatas: Metadata for each chunk
        ids: IDs for each chunk
        embedding_model: Name of the embedding model
    """
    collection = get_or_create_collection(chroma_client, collection_name, embedding_model)
    for metadata in metadatas:
        metadata["timestamp"] = datetime.now().isoformat()
    try:
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
    except Exception as e:
        st.error(f"Error adding to memory: {str(e)}")

def retrieve_from_memory(chroma_client: chromadb.Client, collection_name: str, query: str, embedding_model: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant memory chunks based on query similarity.

    Args:
        chroma_client: ChromaDB client
        collection_name: ChromaDB collection name
        query: User query
        embedding_model: Name of the embedding model
        top_k: Number of results to retrieve

    Returns:
        List of retrieved documents
    """
    collection = get_or_create_collection(chroma_client, collection_name, embedding_model)
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        return [
            {"id": results["ids"][0][i], "text": results["documents"][0][i], "metadata": results["metadatas"][0][i]}
            for i in range(len(results["documents"][0]))
        ]
    except:
        return []
