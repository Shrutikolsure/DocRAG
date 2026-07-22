# DocRAG

A simple document retrieval-augmented generation (RAG) app built with Streamlit, LangChain, and OpenAI embeddings.

## Overview

This project ingests documents from URLs, splits them into chunks, creates a FAISS vector store, and uses a ReAct-style graph to answer user questions over the loaded documents.

## Features

- Streamlit-based question-answer UI
- URL-based document ingestion and chunking
- OpenAI embeddings via `langchain-openai`
- FAISS vector store for similarity search
- Local embedding cache support
- Degraded mode support to run without OpenAI embeddings
- CLI entrypoint with `main.py` for non-Streamlit usage

## Requirements

- Python 3.14+
- `.venv` or virtual environment is strongly recommended

Dependencies are defined in `pyproject.toml`.

Key dependencies:

- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-openai`
- `openai`
- `faiss-cpu`
- `pypdf`
- `python-dotenv`
- `wikipedia`

## Installation

```bash
cd D:/Langchainupdated/DocRAG
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you only have `pyproject.toml`, install with:

```bash
python -m pip install .
```

## Configuration

Create a `.env` file in the repo root with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Optional environment variables:

- `MAX_DOCS_INIT`: maximum number of document chunks to ingest at startup (default: `50`)
- `DEGRADED_MODE`: set to `true` to avoid OpenAI embedding calls and use a simple fallback retriever
- `EMBEDDING_CACHE_PATH`: path for cached embeddings (default: `data/embeddings_cache.pkl`)

## Running the app

### Streamlit UI

```bash
.venv\Scripts\activate
python -m streamlit run streamlit_app.py
```

### Run without OpenAI (degraded mode)

```bash
$env:DEGRADED_MODE = "true"
python -m streamlit run streamlit_app.py
```

### CLI / script mode

```bash
.venv\Scripts\activate
python main.py
```

## Project Structure

- `streamlit_app.py` — Streamlit UI and initialization logic
- `main.py` — Command-line entrypoint for the RAG system
- `pyproject.toml` — Package metadata and dependencies
- `requirements.txt` — Optional requirements file
- `src/config/config.py` — Loader for environment and runtime configuration
- `src/document_ingestion/document_processor.py` — URL/PDF/TXT ingestion and chunking
- `src/vectorstore/Vectorstore.py` — Embedding and FAISS vector store management
- `src/graph_builder/graph_builder.py` — Graph/framing logic for the RAG flow
- `src/nodes/reactnode.py` — ReAct agent behaviour and tool integration
- `src/state/rag_state.py` — Application state management
- `src/nodes/nodes.py` — Node definitions used by the graph

## Notes

- This repo uses a local embedding cache to reduce repeated OpenAI calls.
- If you encounter import or reload issues with Streamlit, stop the app and restart it completely.
- `DEGRADED_MODE=true` is useful for local testing when OpenAI credentials are unavailable.

## Troubleshooting

- If the app fails to initialize, verify `OPENAI_API_KEY` is set and the `.env` file is loaded.
- Remove stale Python cache files if you see strange import behavior:

```powershell
Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force
```

- For embedding-related failures, check `data/embeddings_cache.pkl` and delete it to rebuild.

## License

Add your preferred license here.
