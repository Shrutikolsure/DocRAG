"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Import vectorstore lazily to avoid import-time crashes in Streamlit
        # (some environments may load modules differently during reloads).
        from importlib import import_module
        vs_mod = import_module("src.vectorstore.Vectorstore")
        Vectorstore = getattr(vs_mod, "Vectorstore")

        # Some runtime reloads (Streamlit dev loop) may load an older module
        # object that doesn't expose `SimpleRetriever`. Provide a local
        # fallback implementation so the app remains functional immediately.
        try:
            SimpleRetriever = getattr(vs_mod, "SimpleRetriever")
        except AttributeError:
            class SimpleRetriever:
                """Local fallback retriever: simple token-overlap scoring."""
                def __init__(self, documents):
                    self.documents = documents

                def _score(self, doc, query: str) -> int:
                    import re
                    from collections import Counter
                    q_tokens = re.findall(r"\w+", query.lower())
                    if not q_tokens:
                        return 0
                    doc_text = (getattr(doc, "page_content", "") or "").lower()
                    counts = Counter(re.findall(r"\w+", doc_text))
                    return sum(counts.get(t, 0) for t in q_tokens)

                def get_relevant_documents(self, query: str, k: int = 4):
                    scored = [(self._score(d, query), d) for d in self.documents]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [d for s, d in scored[:k] if s > 0]

                def invoke(self, query: str):
                    return self.get_relevant_documents(query)

        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = Vectorstore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents (limit to configured startup count)
        documents = doc_processor.process_url(urls)
        # Use getattr with a sane default to avoid AttributeError if Config
        # wasn't fully initialized for any reason during reloads.
        max_docs = getattr(Config, "MAX_DOCS_INIT", 50)
        documents = documents[:max_docs]

        # Create vector store (or start in degraded mode)
        # Use getattr for DEGRADED_MODE to avoid AttributeError during reloads
        if getattr(Config, "DEGRADED_MODE", False):
            vector_store.retriever = SimpleRetriever(documents)
        else:
            vector_store.create_vectorstore(documents) # type: ignore
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {type(e).__name__}: {str(e)}")
        st.exception(e)
        if type(e).__name__ == "RateLimitError":
            st.error(
                "OpenAI quota exceeded. Check your API key, billing plan, or reduce the number of documents loaded."
            )
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()