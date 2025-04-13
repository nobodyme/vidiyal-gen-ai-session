from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="YOUR_CLUSTER_URL", 
    api_key="YOUR_QDRANT_API_KEY",
)

print(qdrant_client.get_collections())