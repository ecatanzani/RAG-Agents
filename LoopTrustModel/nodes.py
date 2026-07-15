from state import AgentState
from model_answer import FactCheckedAnswer
from typing import Literal

from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def call_model(model: ChatBedrock):
    def model_inference(state: AgentState) -> AgentState:
        raw_messages = state["messages"]
        
        system_instruction = SystemMessage(content=(
            "You are a precise encyclopedic assistant. Answer ALL of the user's questions. "
            "Provide the actual requested demographic data, numbers, figures, or population statistics "
            "inside the 'requested_metrics' field of the schema. Do not use placeholder phrases; provide real data."
        ))
        
        messages = [system_instruction] + raw_messages
        
        if state.get("error"):
            correction_message = HumanMessage(content=f"CRITICAL VALIDATION FAILURE: {state['error']}. Fix it.")
            messages = messages + [correction_message]
        
        pydantic_response = model.invoke(messages)
        chat_message = AIMessage(content=pydantic_response.model_dump_json())
        
        return {
            "messages": [chat_message], 
            "attempts": state.get("attempts", 0) + 1
        }
    
    return model_inference

def fact_check_critic(state: AgentState) -> Literal["regenerate", "finalize"]:
    """Node/Edge: Acts as the programmatic check to trap hallucinations."""
    
    # 1. Recuperiamo l'ultimo messaggio (che ora è un AIMessage contenente una stringa JSON)
    last_message = state["messages"][-1]
    
    # 2. Convertiamo la stringa JSON del messaggio nell'oggetto Pydantic originale
    try:
        last_output = FactCheckedAnswer.model_validate_json(last_message.content)
    except Exception:
        # Se per qualche motivo il parsing fallisce, forziamo una rigenerazione
        state["error"] = "Failed to parse model output into structured format."
        return "regenerate"
    
    known_hallucination_triggers = ["flux capacitor", "perpetual motion"]
    
    # 3. Ora puoi usare `last_output.extracted_facts` esattamente come prima!
    for fact in last_output.extracted_facts:
        if any(trigger in fact.lower() for trigger in known_hallucination_triggers):
            # Trap the hallucination and update state
            state["error"] = f"Detected unauthorized claim/hallucination in facts: '{fact}'."
            
            # Circuit breaker
            if state.get("attempts", 0) >= 3:
                state["error"] = "Maximum validation attempts reached. Aborting loop."
                return "finalize"
                
            return "regenerate"
            
    return "finalize"