import os
import tempfile
import streamlit as st
from PIL import Image
from core_rag_system import OllamaRAGSystem
from hashlib import md5
from datetime import datetime

def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = {}
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = {}
    if "source_mappings" not in st.session_state:
        st.session_state.source_mappings = {}
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"user_collection_{md5(str(datetime.now()).encode()).hexdigest()[:10]}"

def create_streamlit_app():
    """Create the Streamlit application interface."""
    st.set_page_config(page_title="Memory-Graph RAG System", page_icon="🧠", layout="wide")
    init_session_state()

    with st.sidebar:
        st.title("🧠 Memory-Graph RAG")
        st.write("Chat with documents using memory and knowledge graphs.")

        st.header("Model Configuration")
        llm_model = st.selectbox("LLM Model", ["llama3.2:3b-instruct-q8_0", "mistral", "gemma:7b"], index=0)
        embedding_model = st.selectbox("Embedding Model", ["mxbai-embed-large", "nomic-embed-text"], index=0)

        if "rag_system" not in st.session_state or st.button("Reload Models"):
            with st.spinner("Initializing..."):
                st.session_state.rag_system = OllamaRAGSystem(llm_model, embedding_model)
                st.success("RAG system initialized!")

        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx", "csv", "xlsx"])
        if uploaded_file:
            document_name = st.text_input("Document Name", value=uploaded_file.name)
            if st.button("Process Document"):
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        doc_info = st.session_state.rag_system.process_document(
                            tmp_path, document_name, st.session_state.collection_name
                        )
                        st.session_state.document_chunks[document_name] = doc_info
                        for i, chunk_id in enumerate(doc_info["chunk_ids"]):
                            st.session_state.source_mappings[chunk_id] = {
                                "document": document_name, "text": doc_info["chunks"][i], "metadata": doc_info["metadatas"][i]
                            }
                        st.success(f"Processed '{document_name}' with {len(doc_info['chunks'])} chunks.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                    finally:
                        os.unlink(tmp_path)

        st.header("Graph Visualization")
        graph_size = st.slider("Graph Size", 300, 800, 500, 50)

    st.title("Memory-Based, Graph-Based RAG System")
    st.header("Chat with Documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "graph_image" in message:
                with st.expander("View Details"):
                    st.image(message["graph_image"], caption="Knowledge Graph", width=graph_size)
                    st.markdown("**Sources and Reasoning**:\n" + message.get("details", ""))

    if user_query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, subgraph = st.session_state.rag_system.process_query(user_query, st.session_state.collection_name)
                graph_image = st.session_state.rag_system.visualize_graph(subgraph)
                details = response
                st.markdown(response.split("### Reasoning")[0])
                with st.expander("View Details"):
                    st.image(graph_image, caption="Knowledge Graph", width=graph_size)
                    st.markdown("**Sources and Reasoning**:\n" + details)
                st.session_state.messages.append({
                    "role": "assistant", "content": response, "graph_image": graph_image, "details": details
                })

def main():
    """Run the Streamlit app."""
    create_streamlit_app()

if __name__ == "__main__":
    main()
