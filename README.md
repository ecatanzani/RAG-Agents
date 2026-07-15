# 🪐 Enterprise Gen-AI, RAG & Agentic Ecosystem blueprints

Welcome to the definitive reference repository for building, containerizing, and orchestrating production-ready Generative AI systems. This project serves as an end-to-end blueprint utilizing **LangChain**, **LangGraph**, **ChromaDB**, **FastAPI**, **Streamlit**, and **Amazon Bedrock (Amazon Nova & Titan)**.

## 🏗️ Core Blueprints

### 1. 🧠 LoopTrust Model (`main.py` / `nodes.py`)
An advanced **Agentic AI Workflow** graph designed to mimic an automated writer-critic optimization cycle, explicitly ensuring trustworthy outputs.
* **Architecture:** Uses a non-linear `StateGraph` where generation tasks feed directly into an automated validation node (`fact_check_critic`).
* **Capabilities:** If an anomaly, hallucination, or structural deviation is caught, the graph dynamically increments its global `attempts` tracker and loops back to retry, feeding the precise error context to the LLM for native self-correction. It enforces structured JSON responses via Pydantic schema validation.

### 2. 🔌 SimpleChat Ecoystem (`api.py` & `app.py`)
A unified codebase demonstrating how the same underlying model profile can power distinct interface paradigms depending on downstream architecture needs.
* **FastAPI Microservice Backend (`api.py`):** A headless REST API exposing a `POST /api/v1/chat` endpoint. It leverages an in-memory session ledger to maintain context-isolated multi-turn conversational threads across independent user requests, featuring automated rollback error cleanup.
* **Streamlit Web Application (`app.py`):** A quick, highly responsive conversational web layout implementing resource caching (`@st.cache_resource`) to maintain model connection handshakes across client page reruns.

### 3. 📚 SimpleRAG Engine (`rag_demo.py`)
A lightweight, context-grounded **Retrieval-Augmented Generation** implementation.
* **Architecture:** Splices raw vector transformations across a multi-stage LangChain Expression Language (LCEL) runtime chain:
  ```text
  [Query] ──► [Titan Text Embeddings v2] ──► [Chroma DB (k=2)] ──► [Context Prompt Context] ──► [Nova Lite]