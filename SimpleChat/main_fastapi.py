from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Union
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

app = FastAPI(title="Bedrock Chat API")

try:
    model_id = 'eu.amazon.nova-lite-v1:0'
    model = ChatBedrock(
        model_id=model_id,
        temperature=0.7
    )
except Exception as e:
    print(f"AWS Initialization Error: {e}")
    model = None

SYSTEM_PROMPT = (
    "You are a helpful, brilliant, and slightly witty AI assistant. "
    "Your goal is to help the user solve technical problems, brainstorm ideas, "
    "and write clean code. Keep your responses structured, concise, and engaging."
)

sessions_db: Dict[str, List[Union[SystemMessage, HumanMessage, AIMessage]]] = {}


class ChatRequest(BaseModel):
    session_id: str
    user_query: str


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="AWS Bedrock model not initialized. Check server logs.")

    if request.session_id not in sessions_db:
        sessions_db[request.session_id] = [SystemMessage(content=SYSTEM_PROMPT)]
    
    chat_history = sessions_db[request.session_id]

    try:
        chat_history.append(HumanMessage(content=request.user_query))
        response = model.invoke(chat_history)
        response_text = response.content
        chat_history.append(AIMessage(content=response_text))
        return {
            "session_id": request.session_id,
            "assistant_response": response_text
        }

    except Exception as e:
        if chat_history and isinstance(chat_history[-1], HumanMessage):
            chat_history.pop()
        raise HTTPException(status_code=500, detail=f"Error communicating with AWS Bedrock: {str(e)}")