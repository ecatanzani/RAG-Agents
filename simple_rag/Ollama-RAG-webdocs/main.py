'''
Example of simple RAG-based LLM
Taken from: https://docs.langchain.com/oss/python/langchain/rag#langsmith

We want to create a simple application that takes a user question, searches for documents relevant to that question, passes the retrieved documents and initial question to a model, and returns an answer.

The model should use the tool only if needed (context is important).
'''

import bs4
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

from langchain_community.document_loaders import WebBaseLoader # fmt: off

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='RAG Agent')
    parser.add_argument("-q", "--query", type=str,
                        required=True, dest='user_query', help='User query')
    parser.add_argument("-t", "--temperature", type=float,
                        default=0.0, dest="temperature", help='LLM Temperature')
    parser.add_argument("-d", "--docs", type=str,
                        dest='documentation', default="https://lilianweng.github.io/posts/2023-06-23-agent/",
                        help='Online documentation web address')
    parser.add_argument("--chunk_size", type=int, dest="chunk_size", default=1000, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, dest="chunk_overlap", default=200, help="Chunk overlap")

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
    # Only keep post title, headers, and content from the full HTML.
    bs4_strainer = bs4.SoupStrainer(
        class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(args.documentation,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    assert len(docs) == 1
    logger.info(f"Total characters: {len(docs[0].page_content)}")

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
    def get_dynamic_agent(url: str):
        # Read the documentation
        loader = WebBaseLoader(url)
        docs = loader.load()

        topic_extractor_prompt = PromptTemplate.from_template(
            "You are a metadata extractor. Based on this text, what is the main subject "
            "matter? Answer in 5 words or less.\n\nText: {content}"
        )

        # Direct call to the model to get the topic string
        topic_chain = topic_extractor_prompt | model
        topic_response = topic_chain.invoke({"content": docs[0].page_content[:2000]})
        main_topic = topic_response.content.strip()

        logger.info(f"Detected Topic: {main_topic}")

        # Define the dynamic tool
        def retrieve_context_func(query: str, top_k: int = 5):
            retrieved_docs = vector_store.similarity_search(query, k=top_k)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized

        dynamic_tool = StructuredTool.from_function(
            func=retrieve_context_func,
            name="retrieve_context",
            description=(
                f"A specialized search engine for documentation about: {main_topic}. "
                f"STRICT USE CASE: Only use this tool if the user's question requires "
                f"specific technical details found in {main_topic} documents. "
                f"FORBIDDEN USE CASE: Do not use this for general knowledge, common facts, "
                f"or any topic that is NOT {main_topic}. If you can answer from your own "
                f"internal training without this documentation, do so directly."
            )
        )

        # Construct the Dynamic System Prompt
        dynamic_prompt = (
            f"You are a dual-mode AI expert. Your primary specialized knowledge is in: {main_topic}.\n\n"
            "OPERATING MODES:\n"
            f"1. SPECIALIST MODE: If the query is about {main_topic}, you MUST use 'retrieve_context' "
            "to provide accurate, documentation-backed answers.\n"
            "2. GENERALIST MODE: For any other topic (world history, general tech, lifestyle, "
            "everyday objects), answer directly using your internal memory. DO NOT call the tool.\n\n"
            "DECISION RULE:\n"
            f"Ask yourself: 'Is this question specifically about {main_topic}?'\n"
            "- If YES: Use the tool.\n"
            "- If NO: Answer immediately without the tool."
        )

        return create_agent(model, [dynamic_tool], system_prompt=dynamic_prompt)

    dynamic_agent = get_dynamic_agent(args.documentation)

    # Inference
    for event in dynamic_agent.stream(
        {"messages": [{"role": "user", "content": args.user_query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()

if __name__ == '__main__':
    main()