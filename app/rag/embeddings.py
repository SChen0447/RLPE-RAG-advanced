from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)



