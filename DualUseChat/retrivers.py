from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

docs = [
    Document(
        page_content="Polizza POL-2024-004471: copertura RC auto, massimale 1.000.000 EUR. "
        "Cliente: Rossi Mario. Decorrenza 01/03/2024.",
        metadata={"id": "POL-2024-004471"},
    ),
    Document(
        page_content="Polizza POL-2024-004472: copertura incendio e danni da acqua per "
        "abitazione civile, massimale 300.000 EUR.",
        metadata={"id": "POL-2024-004472"},
    ),
    Document(
        page_content="Le polizze infortuni coprono invalidità permanente e temporanea "
        "derivante da eventi accidentali extraprofessionali.",
        metadata={"id": "GEN-INFORTUNI"},
    ),
    Document(
        page_content="In caso di sinistro con danni causati da acqua o allagamento, "
        "il cliente deve presentare denuncia entro 3 giorni lavorativi.",
        metadata={"id": "PROC-SINISTRI-ACQUA"},
    ),
]

embeddings = BedrockEmbeddings(
    model_id = "amazon.titan-embed-text-v2:0",
    region_name = "eu-south-1"
)

llm = ChatBedrockConverse(
    model_id = "eu.amazon.nova-lite-v1:0",
    region_name = "eu-south-1",
    temperature = 0
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)
retiever = vectorstore.as_retriever(search_kwargs={"k": 3})

query_semantica = "cosa copre l'assicurazione in caso di allagamento in casa?"

print("\n=== Query semantica ===")
for d in retiever.invoke(query_semantica):
    print("-", d.metadata["id"], ":", d.page_content[:60])


results = vectorstore.similarity_search_with_score("your query", k=3)
for doc, score in results:
    print(f"Score: {score}")
    print(f"Content: {doc.page_content}\n")


prompt = ChatPromptTemplate.from_messages(
    (
        "system",
        (
            "Sei un assistente che deve rispondere alla domanda di un cliente su una polizza"
            "CONTESTO"
            "{context}"
        ),
        ("human", "{query}")
    )
)

def postprocess_rag_data(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = {"context": retiever | postprocess_rag_data, "query": RunnablePassthrough()} | prompt | llm | StrOutputParser()

answer = chain.invoke(query_semantica)

print("query", query_semantica)
print("risposta", answer)