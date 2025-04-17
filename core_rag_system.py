from typing import List, Dict, Any, Tuple
import networkx as nx
from PIL import Image
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
import chromadb
from config_prompts import EXTRACT_ENTITIES_TEMPLATE, RAG_TEMPLATE
from core_document_processor import extract_text_from_document
from core_memory_manager import add_to_memory, retrieve_from_memory, get_or_create_collection
from core_graph_manager import extract_and_add_to_graph, retrieve_from_graph, extract_subgraph
from core_utils import format_memory_context, format_graph_context, format_source_references, visualize_graph

class OllamaRAGSystem:
    """Main class implementing the memory-based, graph-based RAG system using Ollama."""

    def __init__(self, llm_model: str = "llama3", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the RAG system with specified models.

        Args:
            llm_model: The name of the Ollama LLM model (default: llama3)
            embedding_model: The name of the Ollama embedding model (default: mxbai-embed-large)
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        # Initialize Ollama for LLM and embeddings
        self.llm = Ollama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Memory representation as knowledge graph
        self.knowledge_graph = nx.DiGraph()

        # Initialize chains
        self.entity_extraction_chain = LLMChain(llm=self.llm, prompt=EXTRACT_ENTITIES_TEMPLATE)
        self.rag_chain = LLMChain(llm=self.llm, prompt=RAG_TEMPLATE)

    def process_document(self, file_path: str, document_name: str, collection_name: str) -> Dict[str, Any]:
        """
        Process a document: extract text, split into chunks, store in ChromaDB, and build knowledge graph.

        Args:
            file_path: Path to document
            document_name: Document name for reference
            collection_name: ChromaDB collection name

        Returns:
            Dictionary with chunks, IDs, and metadata
        """
        import streamlit as st
        try:
            text = extract_text_from_document(file_path)
            chunks = self.text_splitter.split_text(text)
            chunk_ids = [f"doc_{document_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": document_name,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "document_type": file_path.split('.')[-1],
                    "timestamp": datetime.datetime.now().isoformat()
                }
                for i in range(len(chunks))
            ]

            add_to_memory(self.chroma_client, collection_name, chunks, metadatas, chunk_ids, self.embedding_model)

            for i, chunk in enumerate(chunks):
                extract_and_add_to_graph(self.knowledge_graph, chunk, chunk_ids[i], document_name, self.entity_extraction_chain)

            return {"chunks": chunks, "chunk_ids": chunk_ids, "metadatas": metadatas}
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return {}

    def process_query(self, query: str, collection_name: str) -> Tuple[str, nx.DiGraph]:
        """
        Process a user query through the RAG pipeline.

        Args:
            query: User query
            collection_name: ChromaDB collection name

        Returns:
            Tuple of (response, subgraph)
        """
        from datetime import datetime
        retrieved_docs = retrieve_from_memory(self.chroma_client, collection_name, query, self.embedding_model)
        relevant_components = retrieve_from_graph(self.knowledge_graph, query, self.embeddings)
        subgraph = extract_subgraph(self.knowledge_graph, relevant_components)
        memory_context = format_memory_context(retrieved_docs)
        graph_context = format_graph_context(self.knowledge_graph, relevant_components)
        source_references = format_source_references(retrieved_docs, relevant_components, self.knowledge_graph)

        response = self.rag_chain.run(
            query=query, memory_context=memory_context, graph_context=graph_context, source_references=source_references
        )

        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        add_to_memory(
            self.chroma_client,
            collection_name,
            [query, response],
            [
                {"type": "query", "conversation_id": conversation_id, "timestamp": datetime.now().isoformat()},
                {"type": "response", "conversation_id": conversation_id, "timestamp": datetime.now().isoformat()}
            ],
            [f"{conversation_id}_query", f"{conversation_id}_response"],
            self.embedding_model
        )

        return response, subgraph

    def visualize_graph(self, graph: nx.DiGraph) -> Image.Image:
        """
        Visualize the knowledge graph.

        Args:
            graph: NetworkX DiGraph

        Returns:
            PIL Image of visualization
        """
        return visualize_graph(graph)
