'''
Example of simple RAG-based LLM
Taken from: https://docs.langchain.com/oss/python/langchain/rag#langsmith

We want to create a simple application that takes a user question, searches for documents relevant to that question, passes the retrieved documents and initial question to a model, and returns an answer.

The model should use the tool only if needed (context is important).
'''

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from dotenv import load_dotenv
import argparse
import os
import logging
import sys

# Loads .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader # fmt: off

SCORE_THRESHOLD: int = 0.4

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='RAG Agent')
    parser.add_argument("-q", "--query", type=str,
                        required=True, dest='user_query', help='User query')
    parser.add_argument("-t", "--temperature", type=float,
                        default=0.0, dest="temperature", help='LLM Temperature')
    parser.add_argument("-d", "--docs", type=str,
                        required=True, dest='documentation', help='Local PDF folder')
    parser.add_argument("--chunk_size", type=int, dest="chunk_size", default=600, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, dest="chunk_overlap", default=50, help="Chunk overlap")

    args = parser.parse_args()
    return args

def get_logger(name: str, level: int = logging.INFO):
    """
    Creates and returns a configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def main():
    args = parse_args()
    logger = get_logger(name="RAG Pipeline")

    logger.info(f"Tracing enabled: {os.getenv('LANGSMITH_TRACING').capitalize()}")
    logger.info(f"Ollama host: {os.getenv('OLLAMA_HOST')}")
    logger.info(f"Ollama LLM model: {os.getenv('OLLAMA_MODEL')}")
    logger.info(f"Ollama EMBEDDING_MODEL model: {os.getenv('OLLAMA_EMBEDDING_MODEL')}")

    # LLM for generation
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL"),
        base_url=os.getenv("OLLAMA_HOST"),
        temperature=args.temperature,
    )

    # Embeddings model
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
        base_url=os.getenv("OLLAMA_HOST"),
    )

    # Create a vector store
    vector_store = InMemoryVectorStore(embeddings)

    # Loading documents
    loader = PyPDFDirectoryLoader(args.documentation)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from PDFs in {args.documentation}")

    # Splitting documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,  # chunk size (characters)
        chunk_overlap=args.chunk_overlap,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    logger.info(f"Split blog post into {len(all_splits)} sub-documents.")

    # Store documents
    vector_store.add_documents(documents=all_splits)

    # Extract documentation context
    def get_dynamic_agent(docs, vector_store, model, logger):
        # Read the documentation
        sample_content = "\n".join([d.page_content for d in docs[:10]])[:10000]

        topic_extractor_prompt = PromptTemplate.from_template(
            "Analyze the following document excerpts and identify the core subject matter. "
            "ONLY provide FEW words (up to five) title for this topic. Should be plain text, so no symbols. \n\nContent: {content}"
        )

        # Direct call to the model to get the topic string
        topic_chain = topic_extractor_prompt | model
        topic_response = topic_chain.invoke({"content": sample_content})
        main_topic = topic_response.content.strip()

        logger.info(f"Detected Topic: {main_topic}")

        # Define the dynamic tool
        def retrieve_context_func(query: str):
            top_k: int = 5
            retrieved_docs = vector_store.similarity_search_with_score(query, k=top_k)

            # DEBUG
            for doc, score in retrieved_docs:
                logger.info(f">>> [TOOL DEBUG] Score: {score:.4f} | Content: {doc.page_content[:50]}...")

            relevant_docs = [doc for doc, score in retrieved_docs if score < SCORE_THRESHOLD]
            if not relevant_docs:
                return "I don't have relevant information in the knowledge base."
            return "\n\n".join(doc.page_content for doc in relevant_docs)

        dynamic_tool = StructuredTool.from_function(
            func=retrieve_context_func,
            name="retrieve_context",
            description=f"Search the local PDF database about {main_topic}."
        )

        # Construct the Dynamic System Prompt
        dynamic_prompt = (
            f"You are a helpful assistant. You have access to a technical database about: {main_topic}.\n\n"
            f"RULE 1: If the user asks about {main_topic}, use the 'retrieve_context' tool.\n"
            f"RULE 2: If the user asks about anything else not related to {main_topic} (like watches, cooking, or general facts), "
            "ignore the database and answer immediately from your internal memory.\n"
            "DO NOT output JSON. Speak naturally in plain text."
        )

        return create_agent(model, [dynamic_tool], system_prompt=dynamic_prompt)

    dynamic_agent = get_dynamic_agent(docs, vector_store, model, logger)

    # Inference
    for event in dynamic_agent.stream(
        {"messages": [{"role": "user", "content": args.user_query}]},
        stream_mode="updates",
    ):
        # 'event' in modalità updates è un dizionario: { "nome_nodo": { "chiave_stato": valore } }
        for node_name, node_update in event.items():
            # Verifichiamo se il nodo ha effettivamente aggiornato i messaggi
            if "messages" in node_update:
                # Stampiamo l'ultimo messaggio prodotto da questo specifico nodo
                last_msg = node_update["messages"][-1]
                last_msg.pretty_print()

if __name__ == '__main__':
    main()