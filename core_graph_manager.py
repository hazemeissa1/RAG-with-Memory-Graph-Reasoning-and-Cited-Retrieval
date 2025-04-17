from typing import List, Dict, Any
import json
import re
import networkx as nx
import numpy as np
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings

def extract_and_add_to_graph(graph: nx.DiGraph, text: str, chunk_id: str, document_name: str, entity_extraction_chain: Any) -> None:
    """
    Extract entities and relationships and add to knowledge graph.

    Args:
        graph: NetworkX DiGraph
        text: Text to extract from
        chunk_id: Chunk ID
        document_name: Document name
        entity_extraction_chain: LangChain LLMChain for entity extraction
    """
    try:
        result = entity_extraction_chain.run(text=text)
        parsed_result = json.loads(re.search(r'\{.*\}', result, re.DOTALL).group(0))
        entities = parsed_result.get("entities", [])
        relationships = parsed_result.get("relationships", [])

        for entity in entities:
            graph.add_node(
                entity["id"],
                name=entity["name"],
                type=entity["type"],
                source_document=document_name,
                source_chunk=chunk_id
            )

        for rel in relationships:
            graph.add_edge(
                rel["source"],
                rel["target"],
                type=rel["type"],
                description=rel["description"],
                source_document=document_name,
                source_chunk=chunk_id
            )
    except Exception as e:
        st.warning(f"Failed to extract entities for chunk {chunk_id}: {str(e)}")

def retrieve_from_graph(graph: nx.DiGraph, query: str, embeddings: OllamaEmbeddings, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant graph components based on query similarity.

    Args:
        graph: NetworkX DiGraph
        query: User query
        embeddings: Ollama embeddings model
        top_k: Number of results to retrieve

    Returns:
        List of relevant graph components
    """
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    node_texts = {n: f"{d['name']} ({d['type']})" for n, d in graph.nodes(data=True)}
    edge_texts = {
        f"{u}_{v}": f"{node_texts.get(u, u)} {d['type']} {node_texts.get(v, v)}: {d['description']}"
        for u, v, d in graph.edges(data=True)
    }

    all_texts = list(node_texts.values()) + list(edge_texts.values())
    all_ids = list(node_texts.keys()) + list(edge_texts.keys())

    if not all_texts:
        return []

    query_embedding = embeddings.embed_query(query)
    text_embeddings = [embeddings.embed_query(text) for text in all_texts]
    similarities = [cosine_similarity(query_embedding, emb) for emb in text_embeddings]

    relevant_components = []
    for idx in np.argsort(similarities)[::-1][:top_k]:
        component_id = all_ids[idx]
        if component_id in node_texts:
            node_data = graph.nodes[component_id]
            relevant_components.append({
                "type": "node", "id": component_id, "name": node_data["name"],
                "entity_type": node_data["type"], "source_document": node_data["source_document"],
                "source_chunk": node_data["source_chunk"], "similarity": similarities[idx]
            })
        else:
            source, target = component_id.split("_")
            edge_data = graph.get_edge_data(source, target)
            relevant_components.append({
                "type": "edge", "source": source, "target": target,
                "relationship_type": edge_data["type"], "description": edge_data["description"],
                "source_document": edge_data["source_document"], "source_chunk": edge_data["source_chunk"],
                "similarity": similarities[idx]
            })
    return relevant_components

def extract_subgraph(graph: nx.DiGraph, relevant_components: List[Dict[str, Any]], hops: int = 1) -> nx.DiGraph:
    """
    Extract a subgraph with relevant nodes and neighbors.

    Args:
        graph: NetworkX DiGraph
        relevant_components: Relevant graph components
        hops: Number of hops to include

    Returns:
        NetworkX DiGraph
    """
    subgraph = nx.DiGraph()
    relevant_nodes = {c["id"] for c in relevant_components if c["type"] == "node"} | \
                    {c["source"] for c in relevant_components if c["type"] == "edge"} | \
                    {c["target"] for c in relevant_components if c["type"] == "edge"}

    nodes_to_explore = list(relevant_nodes)
    explored = set()
    for _ in range(hops):
        new_nodes = []
        for node in nodes_to_explore:
            if node in explored:
                continue
            explored.add(node)
            if node in graph:
                subgraph.add_node(node, **graph.nodes[node])
                for neighbor in graph.successors(node):
                    new_nodes.append(neighbor)
                    subgraph.add_edge(node, neighbor, **graph.get_edge_data(node, neighbor))
                for neighbor in graph.predecessors(node):
                    new_nodes.append(neighbor)
                    subgraph.add_edge(neighbor, node, **graph.get_edge_data(neighbor, node))
        nodes_to_explore = new_nodes
    return subgraph
