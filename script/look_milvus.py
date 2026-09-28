from pymilvus import MilvusClient
from app.config import MILVUS_URI, MILVUS_COLLECTION_NAME

client = MilvusClient(uri=MILVUS_URI)  
desc = client.describe_collection(MILVUS_COLLECTION_NAME)
for f in desc["fields"]:
    print(f["name"], f["type"])