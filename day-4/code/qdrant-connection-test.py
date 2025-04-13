import os
from qdrant_client import QdrantClient


QDRANT_URL = os.environ.get("QDRANT_CLUSTER_URL")  
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    print("Error: QDRANT_CLUSTER_URL, QDRANT_API_KEY not found in environment variables."
          "Please set them in your .env file or directly in the environment.")
    exit()

qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)

print(qdrant_client.get_collections())