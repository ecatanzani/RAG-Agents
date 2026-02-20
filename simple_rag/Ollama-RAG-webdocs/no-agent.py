import bs4
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import argparse
import os
import logging
import sys

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid RAG Pipeline")
    parser.add_argument("-q", "--query", type=str, required=True, dest="user_query")
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-d", "--docs", type=str,
                        default="https://lilianweng.github.io/posts/2023-06-23-agent/")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=200)
    return parser.parse_args()


def get_logger(name: str, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
    return logger


def main():
    args = parse_args()
    logger = get_logger("Hybrid RAG")

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
        class_=("post-title", "post-header", "post-content")
    )
    loader = WebBaseLoader(
        web_paths=(args.docs,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    logger.info(f"Total characters: {len(docs[0].page_content)}")

    # Splitting documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )

    splits = splitter.split_documents(docs)
    vector_store.add_documents(splits)

    logger.info(f"Indexed {len(splits)} chunks")

    # Main topic extraction
    topic_extractor_prompt = PromptTemplate.from_template(
        "You are a metadata extractor. Based on this text, what is the main subject "
        "matter? Answer in 5 words or less.\n\nText: {content}"
    )
    topic_chain = topic_extractor_prompt | model
    topic_response = topic_chain.invoke({"content": docs[0].page_content[:2000]})
    main_topic = topic_response.content.strip()
    logger.info(f"Detected Topic: {main_topic}")


    # Quesry classification
    classifier_prompt = PromptTemplate.from_template(
        """
You are a query router.

Documentation topic: {topic}

User question: {question}

Answer ONLY with:
- "DOC" if the question is specifically about the documentation topic.
- "GEN" if it is general knowledge or unrelated.
"""
    )

    classifier_chain = classifier_prompt | model

    classification = classifier_chain.invoke({
        "topic": main_topic,
        "question": args.user_query
    }).content.strip()

    logger.info(f"Query classified as: {classification}")

    # Routing
    if classification == "DOC":
        logger.info("Running RAG pipeline...")

        retrieved_docs = vector_store.similarity_search(
            args.user_query,
            k=5
        )

        context = "\n\n".join(
            doc.page_content for doc in retrieved_docs
        )

        rag_prompt = f"""
You are a documentation expert.

Use ONLY the following context to answer.

Context:
{context}

Question:
{args.user_query}
"""

        response = model.invoke(rag_prompt)
        print("\nFinal Answer:\n")
        print(response.content)

    else:
        logger.info("Running direct LLM response...")

        response = model.invoke(args.user_query)
        print("\nFinal Answer:\n")
        print(response.content)


if __name__ == "__main__":
    main()