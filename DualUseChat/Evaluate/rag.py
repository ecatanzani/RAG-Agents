import argparse
from typing import Tuple
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

CHROMA_PATH: str = "../chromadb-data"
CHROMA_COLLECTION_NAME: str = "dual_use_collection"
AWS_EMBEDDINGS_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
AWS_EMBEDDINGS_REGION: str = "eu-south-1"
AWS_LLM_MODEL_ID: str = "eu.amazon.nova-pro-v1:0"
AWS_LLM_REGION: str = "eu-south-1"
AWS_LLM_TEMPERATURE: float = 0
N_RETRIEVED_DOCS: int = 10

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG system")
    parser.add_argument(
        "--question",
        required=True,
        help="User question",
    )
    return parser.parse_args()

class ModelAnswer(BaseModel):
    answer: str = Field(description="Risposta alla domanda dell'utente basata sul contesto.")
    exact_quote: str = Field(description="La citazione esatta dal testo che supporta la risposta. Se non la trovi, scrivi 'N/A'.")

def post_proc_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def answer_question(user_question: str) -> Tuple[str, str]:
    embeddings_model = BedrockEmbeddings(
        model_id=AWS_EMBEDDINGS_MODEL_ID,
        region_name=AWS_EMBEDDINGS_REGION
    )

    # Semantic vectorstore and retriever
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": N_RETRIEVED_DOCS})

    # Add BM25 retriever
    chroma_data = vector_store.get()
    all_texts = chroma_data['documents']
    
    bm25_retriever = BM25Retriever.from_texts(all_texts)
    bm25_retriever.k = N_RETRIEVED_DOCS

    # Combine the retrievers
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5] 
    )

    llm = ChatBedrockConverse(
        model_id=AWS_LLM_MODEL_ID,
        temperature=AWS_LLM_TEMPERATURE,
        region_name=AWS_LLM_REGION
    )
    structured_llm = llm.with_structured_output(ModelAnswer)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", (
                "Sei un assistente virtuale legale e tecnico professionista..\n"
                "Il tuo obiettivo è rispondere alle domande basandoti rigorosamente sulle clausole e sui documenti forniti.\n\n"
                "REGOLA CRITICA\n"
                "Rispondi alla domanda dell'utente utilizzando ESCLUSIVAMENTE il contesto del documento fornito qui sotto. "
                "Mantieni un'assoluta precisione legale. "
                "Se la risposta o un dettaglio specifico non possono essere trovati nel contesto, dichiara esplicitamente: 'Non sono in grado di verificarlo sulla base della documentazione fornita."
                "Non estrapolare o speculare oltre il testo.\n\n"
                "--- PROVIDED DOCUMENT CONTEXT ---\n"
                "{context_str}"
            )
        ),
        ("user", "{user_query}")
    ])

    rag_chain = (
        {"context_str": retriever | post_proc_docs, "user_query": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    answer = rag_chain.invoke(user_question)

    final_answer = answer.answer
    retrieved_context = answer.exact_quote

    return final_answer, retrieved_context

def main():
    args = parse_args()
    user_question = args.question

    embeddings_model = BedrockEmbeddings(
        model_id=AWS_EMBEDDINGS_MODEL_ID,
        region_name=AWS_EMBEDDINGS_REGION
    )

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": N_RETRIEVED_DOCS})

    llm = ChatBedrockConverse(
        model_id=AWS_LLM_MODEL_ID,
        temperature=AWS_LLM_TEMPERATURE,
        region_name=AWS_LLM_REGION
    )
    structured_llm = llm.with_structured_output(ModelAnswer)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", (
                "Sei un assistente virtuale legale e tecnico professionista..\n"
                "Il tuo obiettivo è rispondere alle domande basandoti rigorosamente sulle clausole e sui documenti forniti.\n\n"
                "REGOLA CRITICA\n"
                "Rispondi alla domanda dell'utente utilizzando ESCLUSIVAMENTE il contesto del documento fornito qui sotto. "
                "Mantieni un'assoluta precisione legale. "
                "Se la risposta o un dettaglio specifico non possono essere trovati nel contesto, dichiara esplicitamente: 'Non sono in grado di verificarlo sulla base della documentazione fornita."
                "Non estrapolare o speculare oltre il testo.\n\n"
                "--- PROVIDED DOCUMENT CONTEXT ---\n"
                "{context_str}"
            )
        ),
        ("user", "{user_query}")
    ])

    rag_chain = (
        {"context_str": retriever | post_proc_docs, "user_query": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    answer = rag_chain.invoke(user_question)

    print(f'Model answer: {answer.answer}')
    print(f'Model context: {answer.exact_quote}')

    retrieved_docs = retriever.invoke(user_question)

    print("\n--- DEBUG: COSA HA TROVATO IL RETRIEVER ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[Documento {i+1}]:\n{doc.page_content[:200]}...") # Stampa le prime 200 lettere
    print("--------------------------------------\n")


if __name__ == '__main__':
    main()