from typing import List, Dict, Any
import datetime
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import io

def format_memory_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Format retrieved memory documents.

    Args:
        retrieved_docs: Retrieved documents

    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return "No relevant memory found."
    return "\n".join(
        f"Memory {i+1} - From {d['metadata']['source']} ({format_relative_time(d['metadata']['timestamp'])}):\n{d['text']}\n"
        for i, d in enumerate(retrieved_docs)
    )

def format_relative_time(timestamp_str: str) -> str:
    """
    Format timestamp as relative time.

    Args:
        timestamp_str: ISO timestamp

    Returns:
        Relative time string
    """
    try:
        timestamp = datetime.datetime.fromisoformat(timestamp_str)
        delta = datetime.datetime.now() - timestamp
        if delta.days == 0:
            return "today"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 30:
            return f"{delta.days // 7} weeks ago"
        else:
            return f"{delta.days // 30} months ago"
    except:
        return "unknown time"

def format_graph_context(graph: nx.DiGraph, relevant_components: List[Dict[str, Any]]) -> str:
    """
    Format graph components.

    Args:
        graph: NetworkX DiGraph
        relevant_components: Relevant graph components

    Returns:
        Formatted context string
    """
    if not relevant_components:
        return "No relevant graph information found."
    parts = ["Relevant Concepts:"]
    for c in [c for c in relevant_components if c["type"] == "node"]:
        parts.append(f"  - {c['name']} (Type: {c['entity_type']})")
    parts.append("\nRelevant Relationships:")
    for c in [c for c in relevant_components if c["type"] == "edge"]:
        source_name = graph.nodes.get(c["source"], {}).get("name", c["source"])
        target_name = graph.nodes.get(c["target"], {}).get("name", c["target"])
        parts.append(f"  - {source_name} → {c['relationship_type']} → {target_name}")
        parts.append(f"    Description: {c['description']}")
    return "\n".join(parts)

def format_source_references(retrieved_docs: List[Dict[str, Any]], relevant_components: List[Dict[str, Any]], graph: nx.DiGraph) -> str:
    """
    Format source references.

    Args:
        retrieved_docs: Retrieved documents
        relevant_components: Graph components
        graph: NetworkX DiGraph

    Returns:
        Formatted references string
    """
    sources = {}
    for doc in retrieved_docs:
        meta = doc["metadata"]
        source_key = f"{meta['source']}_chunk_{meta['chunk_index']}"
        sources[source_key] = {
            "name": meta["source"], "type": "document", "chunk_index": meta["chunk_index"],
            "text": doc["text"][:200] + ("..." if len(doc["text"]) > 200 else "")
        }
    for comp in relevant_components:
        source_key = f"{comp['source_document']}_{comp['source_chunk']}"
        if source_key not in sources:
            sources[source_key] = {
                "name": comp["source_document"], "type": "knowledge_graph", "chunk_id": comp["source_chunk"]
            }

    if not sources:
        return "No specific sources identified."
    parts = []
    for i, s in enumerate(sources.values()):
        if s["type"] == "document":
            parts.append(f"Source {i+1}: Document '{s['name']}', Section {s['chunk_index'] + 1}")
            parts.append(f"Excerpt: {s['text']}\n")
        else:
            parts.append(f"Source {i+1}: Knowledge Graph from '{s['name']}', Chunk {s['chunk_id']}\n")
    return "\n".join(parts)

def visualize_graph(graph: nx.DiGraph) -> Image.Image:
    """
    Visualize the knowledge graph.

    Args:
        graph: NetworkX DiGraph

    Returns:
        PIL Image of visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    if not graph.nodes:
        ax.text(0.5, 0.5, "No graph data to display", ha='center', va='center')
        ax.axis('off')
    else:
        pos = nx.spring_layout(graph, seed=42)
        node_colors = [
            {'person': 'skyblue', 'organization': 'lightgreen', 'concept': 'lightcoral'}.get(
                graph.nodes[n].get('type', 'unknown'), 'lightgray'
            ) for n in graph.nodes
        ]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=800, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, arrowsize=15, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, {n: graph.nodes[n]['name'] for n in graph.nodes}, font_size=8)
        nx.draw_networkx_edge_labels(
            graph, pos, {(u, v): d['type'] for u, v, d in graph.edges(data=True)}, font_size=7
        )
        plt.axis('off')
        plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img
