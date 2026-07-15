# 🧠 Self-Correcting Agentic Workflow with Amazon Nova & LangGraph

This application implements an **Agentic AI Workflow** using **LangGraph** and **Amazon Bedrock (Amazon Nova)**. It uses an iterative, loop-based execution structure to generate answers, automatically critique them for hallucinations or missing elements, and self-correct dynamically before producing structured outputs.

---

## 🛠️ Key Capabilities

* **Structured Output Enforcement:** Leverages `with_structured_output` on the `amazon-nova-lite` model to ensure data strictly adheres to a predetermined schema (`FactCheckedAnswer`), handling requested metrics and logical reasoning extraction cleanly.
* **Deterministic Critique Loop:** Features conditional routing via a standard Critic Node. If an anomaly, formatting error, or hallucination is detected, the graph automatically loops back to retry, passing the error state with it.
* **Resilient Failure Mitigation:** Manages execution `attempts` directly within the graph's global state to prevent infinite loops while tracking structural health.
* **State Persistence:** Implements a robust state history (`AgentState`) containing the conversation memory thread, global attempt counters, and execution errors.

---

## 📐 Architecture & Routing Reasoning

The workflow is mapped out as an execution state graph rather than a sequential pipeline. This allows the agent to modify its execution path based on the outcome of the model evaluation.

### Workflow Graph Flowchart

```text
    [ START ]
        │
        ▼
 ┌──────────────┐
 │  inference   │◄─────────┐ (Route: "regenerate")
 └──────┬───────┘          │
        │                  │
        ▼                  │
 ┌──────────────┐          │
 │  fact-check  ├──────────┘
 │   (Critic)   │
 └──────┬───────┘
        │
        │ (Route: "finalize")
        ▼
     [ END ]