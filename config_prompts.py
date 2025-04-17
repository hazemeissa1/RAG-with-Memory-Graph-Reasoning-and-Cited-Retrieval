from langchain.prompts import PromptTemplate

EXTRACT_ENTITIES_TEMPLATE = PromptTemplate(
    input_variables=["text"],
    template="""
    Extract key entities, concepts, and their relationships from the following text. 
    Return a JSON object with:
    - "entities": List of entities with unique IDs, names, and types (e.g., person, concept, organization).
    - "relationships": List of relationships with source/target entity IDs, type, and description.
    
    Example:
    {
        "entities": [
            {"id": "e1", "name": "AI", "type": "concept"},
            {"id": "e2", "name": "Machine Learning", "type": "concept"}
        ],
        "relationships": [
            {"source": "e1", "target": "e2", "type": "subfield", "description": "Machine Learning is a subfield of AI."}
        ]
    }

    Text:
    {text}

    JSON Output:
    """
)

RAG_TEMPLATE = PromptTemplate(
    input_variables=["query", "memory_context", "graph_context", "source_references"],
    template="""
    You are an intelligent assistant with access to memory and a knowledge graph to provide accurate, context-aware responses.

    ### User Query
    {query}

    ### Retrieved Memory Context
    {memory_context}

    ### Retrieved Knowledge Graph Context
    {graph_context}

    ### Source References
    {source_references}

    Provide a detailed response with the following sections:
    1. **Answer**: Directly address the query with a clear, concise response.
    2. **Reasoning**: Explain your thought process, detailing how memory and graph data were used to formulate the answer. Reference specific pieces of information and show logical connections.
    3. **Sources**: Cite the exact sources (document name, chunk index, or graph component) used, including excerpts where applicable.

    Ensure the response is accurate, cites sources explicitly, and explains reasoning clearly.
    """
)
