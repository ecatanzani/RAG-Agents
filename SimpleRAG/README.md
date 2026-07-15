# 📚 Production-Ready RAG System with Amazon Bedrock & ChromaDB

This application demonstrates a modular **Retrieval-Augmented Generation (RAG)** pipeline leveraging **LangChain**, **ChromaDB**, and **Amazon Bedrock**. It embeds a localized knowledge base, stores it in an ephemeral vector repository, and uses an identity-preserving prompt template with **Amazon Nova Lite** to fulfill strict, context-bound query engineering.

---

## 💡 Core Capabilities

* **Dual Bedrock Optimization:** Pairs the ultra-efficient `eu.amazon.titan-embed-text-v2:0` model for processing vector semantics with `eu.amazon.nova-lite-v1:0` running at a low temperature (`0.2`) to ensure deterministic, hallucination-free generation.
* **Smart Text Chunking:** Employs a `RecursiveCharacterTextSplitter` configured with a chunk size of `500` characters and a `50` character sliding-window overlap to preserve semantic context across chunk borders.
* **Ephemeral Vector Topography:** Instantiates an isolated vector database inside a secure, host-managed runtime directory via Python’s `tempfile.mkdtemp()`. This avoids local storage leakages in serverless or containerized deployment architectures.
* **Strict Grounding Enforcement:** Implements a localized system prompt engineering directive that mandates the model fallback to `"I don't know"` if the context bounds do not explicitly support the target query inference.

---

## 📐 Architecture & RAG Pipeline Flow

The workflow relies on LangChain's declarative **Expression Language (LCEL)** to construct an end-to-end data pipeline.

```text
    [ Raw Query ]
         │
         ├──► (RunnablePassthrough) ──────────────────────────────┐
         │                                                        ▼
         └──► [ Retriever ] ──► [ Similarity Search (k=2) ] ──► [ prompt ] ──► [ LLM ] ──► [ Output Parser ]