from app.rag.embeddings import BGEEmbeddings

from pathlib import Path

from langchain_milvus import Milvus
from langchain_milvus.function import BM25BuiltInFunction

from app.config import EMBEDDING_MODEL, K, MILVUS_URI, MILVUS_COLLECTION_NAME


class VectorStore:
    def __init__(
            self, 
            uri: str = MILVUS_URI,
            collection_name: str = MILVUS_COLLECTION_NAME
    ):
        self.uri = uri
        self.collection_name = collection_name
        self.dense_embeddings = BGEEmbeddings(model_name=EMBEDDING_MODEL)
        self._vectordb: Milvus|None = None


    def build(self, chunks, force_rebuild: bool = False):
        """构建或重建数据库"""
        if force_rebuild:
            from pymilvus import connections, utility
            connections.connect(uri=self.uri)
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)

        self._vectordb = Milvus.from_documents(
            documents = chunks,
            embedding = self.dense_embeddings,
            builtin_function=BM25BuiltInFunction(output_field_names="sparse_embedding"),  # 内置BM25函数
            vector_field=["dense_embedding", "sparse_embedding"],
            collection_name=self.collection_name,
            connection_args={"uri": self.uri},
            index_params=[            # 索引参数,有两个索引,一个是HNSW,一个是BM25索引，有几个索引对应几个搜索参数
                {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
                {
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "BM25",
                },
            ],
            search_params=[               # 搜索参数
                {
                    "metric_type": "COSINE",
                    "params": {"ef": 64},
                },
                {
                    "metric_type": "BM25",
                    "params": {},
                },
            ],
            drop_old = False
        )

    def load(self):
        """加载数据库"""
        self._vectordb = Milvus(
            embedding_function = self.dense_embeddings,
            builtin_function=BM25BuiltInFunction(output_field_names="sparse_embedding"),
            vector_field=["dense_embedding", "sparse_embedding"],
            collection_name=self.collection_name,
            connection_args={"uri": self.uri},
            search_params=[               # 加载数据库的时候也必须指定，因为不会自动记录搜索参数
                {
                    "metric_type": "COSINE",
                    "params": {"ef": 64},
                },
                {
                    "metric_type": "BM25",
                    "params": {},
                },
            ],
        )
    
    def get_vectorstore(self):
        if self._vectordb is None:
            raise ValueError("向量数据库未加载，请先调用 load() 方法加载数据库。")
        return self._vectordb

    def search(self, question, k: int = K):
        """检索"""
        if self._vectordb is None:
            raise ValueError("向量数据库未加载，请先调用 load() 方法加载数据库。")
        return self._vectordb.similarity_search(question, k=k, search_kwargs={"ranker_type": "rrf"})

    

if __name__ == "__main__":

    question = "什么是大语言模型？"
    vectordb = VectorStore()
    vectordb.load()
    results = vectordb.search(question, k=3)        

    for i, sim_doc in enumerate(results):
        print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

