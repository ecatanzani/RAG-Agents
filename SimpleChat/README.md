# 🤖 AWS Bedrock Conversational Ecosystem

This repository showcases a dual-interface implementation designed to interact with **Amazon Bedrock's Nova Lite** model. 

It contains two production-ready application architectures:
1. 🔌 **A headless REST API** built with **FastAPI** (for integration with backend systems, mobile apps, or web services).
2. 💻 **An interactive Chat Web Application** built with **Streamlit** (for immediate human-in-the-loop interaction, prototyping, and manual testing).

---

## 💡 Core Capabilities

### 1. Advanced LLM Orchestration via AWS
* **Amazon Nova Lite Support:** Configured to leverage AWS's efficient `eu.amazon.nova-lite-v1:0` model optimized for high-speed, cost-effective, and accurate reasoning.
* **Persona Enforcement:** Instantiates a witty, brilliant, and technical system prompt on every session startup to guide the assistant's behavior, keeping outputs concise and code-focused.

### 2. Conversational Memory & Context Management
* **State Preservation:** Keeps track of chat histories (interleaved system, human, and AI messages) to allow multi-turn conversations where the model remembers preceding context.
* **Message Class Separation:** Utilizes explicit Pydantic-friendly LangChain core primitives (`HumanMessage`, `SystemMessage`, `AIMessage`) to structure prompt context perfectly for the model interface.

---

## ⚡ Application Interfaces

### 🔌 FastAPI Backend (`api.py`)
Provides a highly scalable, stateless HTTP server suitable for microservice-oriented architectures.

* **Capabilities:**
  * **Endpoint:** `POST /api/v1/chat`
  * **Session isolation:** Uses an in-memory dictionary `sessions_db` to maintain independent chat contexts across multiple parallel users mapped to a specific `session_id`.
  * **Fault Tolerance & Cleanup:** If the Bedrock service encounters an error (e.g., rate limits or credentials issue), the API automatically rolls back the user's latest query from the session memory to prevent context corruption on subsequent retries.
  * **Interactive Docs:** Generates automated OpenAPI (`/docs`) and ReDoc (`/redoc`) documentation endpoints out-of-the-box.

#### API Payload Example:
```json
// POST /api/v1/chat
{
  "session_id": "user-session-1234",
  "user_query": "Explain quantum entanglement in one short sentence."
}
```

#### Run the backend with Uvicorn

```bash
uvicorn main_fastapi:app --reload
```

#### Example of sending a post

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "test-session-123",
       "user_query": "Explain quantum computing in one short sentence."
     }'
```