# RAG Evaluation System con LangSmith e AWS Bedrock

Questo repository contiene una pipeline completa per la gestione, l'esecuzione e la valutazione di un sistema **RAG (Retrieval-Augmented Generation)** specializzato in ambito legale e tecnico (in particolare per normative *dual-use*). 

Il sistema sfrutta **LangChain**, utilizza **Amazon Bedrock (Amazon Nova)** sia per la generazione delle risposte che per il ruolo di "LLM-as-a-Judge", si appoggia a **ChromaDB** come vector store locale e monitora le metriche di accuratezza tramite **LangSmith**.

## 🏗️ Architettura del Sistema

Il progetto è suddiviso in tre moduli principali che lavorano in sequenza:
1. **Ingestione Dataset (`upload_dataset_to_langsmith.py`)**: Carica file di test in formato CSV direttamente su LangSmith per creare i benchmark di valutazione.
2. **Motore RAG (`rag.py`)**: Interroga il database vettoriale locale (ChromaDB) usando gli embeddings di AWS Bedrock e genera risposte strutturate con una precisione strettamente ancorata al contesto.
3. **Valutazione (`llm-as-a-judge.py`)**: Esegue i test automatici sul dataset confrontando le risposte del RAG con le risposte di riferimento (Ground Truth) utilizzando un LLM indipendente come giudice.

---

## 🛠️ Prerequisiti e Setup

### 1. Variabili d'Ambiente
Crea un file `.env` nella root del progetto con le tue credenziali AWS e le chiavi di LangSmith:

```env
# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGSMITH_API_KEY="your-langsmith-api-key"

# AWS Configuration (Assicurati che l'utente/ruolo abbia i permessi per Bedrock)
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_DEFAULT_REGION="eu-south-1"