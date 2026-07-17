from langchain_aws import BedrockEmbeddings

from params import (
    AWS_EMBEDDINGS_MODEL_ID,
    AWS_EMBEDDINGS_REGION
)

embeddings_model = BedrockEmbeddings(
    model_id=AWS_EMBEDDINGS_MODEL_ID, 
    region_name=AWS_EMBEDDINGS_REGION
)