import os
from dotenv import load_dotenv

load_dotenv()


# 路径配置
FOLDER_PATH = "data_base/knowledge_db"
VECTOR_DB_PATH = "data_base/vector_db/chroma"


# 文本切分
CHUNK_SIZE = 500
OVERLAP_SIZE = 50


# 嵌入模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# milvus
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION_NAME = "advanced_rag_db"

# 最终返回给模型的文档数量
K = 3

# DeepSeek LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"

TEMPERATURE = 0.7

# Session
MAX_TURNS = 10


