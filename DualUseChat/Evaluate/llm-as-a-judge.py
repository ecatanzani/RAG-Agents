from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag import answer_question

AWS_LLM_JUDGE_MODEL_ID: str = "eu.amazon.nova-pro-v1:0"
AWS_LLM_REGION: str = "eu-south-1"
AWS_LLM_TEMPERATURE: float = 0
DATASET_NAME = "dualuse" 

evaluator_llm = ChatBedrockConverse(
    model_id=AWS_LLM_JUDGE_MODEL_ID,
    temperature=AWS_LLM_TEMPERATURE,
    region_name=AWS_LLM_REGION
)

# RAG wrapper
def predict_rag(inputs: dict) -> dict:
    """
    This function wraps your RAG system. LangSmith will pass the 'question' 
    from your dataset into this function.
    """
    question = inputs["question"]
    answer, context = answer_question(question)
    return {
        "answer": answer,
        "context": context
    }

# Define the structured output format for the judge
class EvaluationGrade(BaseModel):
    score: int = Field(description="Score of 1 for Pass, 0 for Fail")
    reasoning: str = Field(description="Brief explanation for the given score")

def correctness_evaluator(run, example) -> dict:
    """
    Evaluates Correctness: Does the generated answer match the 
    reference answer from your uploaded dataset?
    """
    llm = evaluator_llm.with_structured_output(EvaluationGrade)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert grading assistant. Compare the Student Answer against the Reference Answer. If the Student Answer conveys the same correct information, give a score of 1. If it is incorrect or contradicts the Reference Answer, give a score of 0."),
        ("user", "Question: {question}\n\nReference Answer: {reference}\n\nStudent Answer: {student}")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "question": example.inputs["question"],
        "reference": example.outputs["answer"],
        "student": run.outputs["answer"]
    })
    
    return {"key": "Answer Correctness", "score": result.score, "comment": result.reasoning}

def faithfulness_evaluator(run, example) -> dict:
    """
    Evaluates Faithfulness (Hallucination check): Is the generated 
    answer entirely supported by the retrieved context?
    """
    llm = evaluator_llm.with_structured_output(EvaluationGrade)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert grading assistant. Determine if the Student Answer is fully supported by the Retrieved Context. If it contains facts not present in the context (hallucinations), give a score of 0. If it is fully grounded, give a score of 1."),
        ("user", "Retrieved Context: {context}\n\nStudent Answer: {student}")
    ])
    
    chain = prompt | llm
    
    result = chain.invoke({
        "context": run.outputs["context"],
        "student": run.outputs["answer"]
    })
    
    return {"key": "Faithfulness", "score": result.score, "comment": result.reasoning}

if __name__ == "__main__":
    load_dotenv()
    client = Client()

    print(f"Starting evaluation on dataset: {DATASET_NAME}...")
    
    experiment_results = evaluate(
        predict_rag,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, faithfulness_evaluator],
        experiment_prefix="rag-judge-eval",
        metadata={"version": "1.0", "model": "amazon-nova"}
    )
    
    print("Evaluation complete! Check your LangSmith dashboard to view the results.")