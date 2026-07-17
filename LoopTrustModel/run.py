from typing import List
from langchain_aws.chat_models import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

import argparse
from nodes import (
    call_model,
    fact_check_critic
)
from state import AgentState
from model_answer import FactCheckedAnswer

def parse_args():
    parser = argparse.ArgumentParser(description="Ask questions")
    parser.add_argument('-i', '--input', type=str, required=True,
                        dest="input", help="Input question")
    return parser.parse_args()

def main():
    args = parse_args()
    llm = ChatBedrock(
        model_id="eu.amazon.nova-lite-v1:0",
        model_kwargs={"temperature": 0.0}
    )
    structured_llm = llm.with_structured_output(FactCheckedAnswer)
    workflow = StateGraph(AgentState)
    workflow.add_node("inference", call_model(structured_llm))
    workflow.add_edge(START, "inference")
    workflow.add_conditional_edges(
        "inference",
        fact_check_critic,
        {
            "regenerate": "inference",
            "finalize": END
        }
    )
    app = workflow.compile()
    
    initial_input = {
        "messages": [HumanMessage(content=args.input)],
        "attempts": 0,
        "error": ""
    }
    
    print("🚀 Invoking the graph...")
    final_state = app.invoke(initial_input)
    print("\n🏁 Workflow completed successfully!")
    print("\nTotal execution attempts:", final_state.get("attempts"))
    
    final_message = final_state["messages"][-1]
    
    final_structured_data = FactCheckedAnswer.model_validate_json(final_message.content)
    
    print(f"\nFinal Structured Answer: {final_structured_data.final_answer}")
    print(f"Dati Popolazione: {final_structured_data.requested_metrics}")
    print(f"Reasoning: {final_structured_data.extracted_facts}")

if __name__ == '__main__':
    main()


