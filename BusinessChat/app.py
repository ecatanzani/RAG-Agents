from typing import List
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from params import (
    AWS_LLM_MODEL_ID,
    AWS_LLM_REGION
)
from db import technical_retriever, style_retriever

class AnswerSchema(BaseModel):
    opening: str = Field(description="Standard opening greeting addressed to the customer (e.g., Dear Customer, ...)")
    tone: str = Field(description="Initial sentence of apology or thanks, matching the style of the reference text.")
    technical_instructions: List[str] = Field(description="Bullet points containing precise technical instructions taken from the manual documents.")
    closing: str = Field(description="Standard email closing formula and company signature.")

llm = ChatBedrockConverse(
    model_id=AWS_LLM_MODEL_ID,
    region_name=AWS_LLM_REGION,
    temperature=0
)
structured_llm = llm.with_structured_output(AnswerSchema)


template_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a second-level corporate technical support virtual assistant.\n"
        "Your goal is to generate a professional response for the end customer, "
        "which will then be reviewed by a human operator.\n\n"
        
        "To do this, you must follow two fundamental rules:\n"
        "1. Include ONLY the precise technical information extracted from the provided TECHNICAL DOCUMENTS.\n"
        "2. Faithfully copy the writing style, courtesy, and structure of the text in the REFERENCE EMAIL.\n\n"
        
        "--- TECHNICAL DOCUMENTS (Use this information for the solution) ---\n"
        "{t_docs}\n\n"
        
        "--- REFERENCE EMAIL (Copy this writing style) ---\n"
        "{ref_email}\n"
    )),
    ("user", "Generate the response email for the following customer request: {query}")
])


def postprocess_rag_data(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def handle_support_request(customer_query: str) -> AnswerSchema:
    rag_chain = (
        {
            "t_docs": technical_retriever | postprocess_rag_data,
            "ref_email": style_retriever | postprocess_rag_data,
            "query": RunnablePassthrough()
        }
        | template_prompt
        | structured_llm
    )
    validated_response = rag_chain.invoke(customer_query)
    return validated_response


if __name__ == "__main__":
    test_query = "Good morning, my machine has stopped working and 'Error E1' is displayed on the screen. Can you help me?"
    response = handle_support_request(test_query)
    
    print('response', response)