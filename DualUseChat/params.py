# AWS
AWS_EMBEDDINGS_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
AWS_LLM_MODEL_ID: str = "eu.amazon.nova-lite-v1:0"
AWS_LLM_REGION: str = "eu-south-1"
AWS_LLM_TEMPERATURE: float = 0
AWS_EMBEDDINGS_REGION: str = "eu-south-1"

# Data
DOCUMENTS_DIR: str = "./data"
VECTOR_STORE_DIR: str = "./chromadb-data"
VECTOR_STORE_COLLECTION_NAME = "dual_use_collection"