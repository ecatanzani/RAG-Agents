from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import tempfile

from knowledge import KNOWLEDGE_BASE

class AnswerSchema(BaseModel):
    """The structured answer to the user's question based on the provided context."""
    
    answer: str = Field(
        description="The concise answer to the question. If you don't know the answer, say 'I don't know'."
    )
    sources: list[str] = Field(
        default=[],
        description="List of specific sources or facts extracted from the context used to form the answer."
    )

embeddings_model = BedrockEmbeddings(
    model_id="eu.amazon.titan-embed-text-v2:0", 
    region_name="eu-south-1"
)

llm = ChatBedrockConverse(
    model_id="eu.amazon.nova-lite-v1:0",
    region_name="eu-south-1",
    temperature=0
)
structured_llm = llm.with_structured_output(AnswerSchema)

def create_kb():
    """Create a vector store from knowledge base."""

    # split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )

    chunks = splitter.split_documents([doc])

    # create a vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

def rag():
    """Simple RAG"""

    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:

        {context}

        Question: {question}

        Answer:


        Make sure to answer in a concise manner, 
        and if you don't know the answer, just say "I don't know."""
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    # Test the RAG chain
    # Test
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")