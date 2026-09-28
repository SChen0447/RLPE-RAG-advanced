from app.config import K
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import EnsembleRetriever
from functools import lru_cache


@lru_cache(maxsize= 1) # reranker 只加载一次 
def _get_reranker(): 
    """重排序模型单例，避免重复加载"""
    return CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"),
        top_n=K,
    )


# 使用milvus原生混合检索 + CrossEncoder重排序
def build_hybrid_retriever(vectordb):
    """BM25 + Embedding混合检索器 + CrossEncoder重排序"""
    # 拿到的直接就是混合检索结果
    vector_retriever = vectordb.as_retriever(search_kwargs={"ranker_type": "rrf"})

    cross_encoder_reranker = _get_reranker()

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=vector_retriever,
        base_compressor=cross_encoder_reranker,
    )
    
    return compression_retriever

# 不走重排序CrossEncoder重排序————multi-query子问题使用
def build_raw_retriever(vectordb):
    """BM25 + Embedding原始检索器"""
    return vectordb.as_retriever(search_kwargs={"ranker_type": "rrf"})

def rerank_cross_encoder(query, documents, k):
    """使用CrossEncoder重排序"""
    cross_encoder_reranker = _get_reranker()
    return cross_encoder_reranker.compress_documents(documents, query)[:k]


# 手搓混合检索器 + CrossEncoder重排序
def build_hybrid_retriever_intelligence(vectordb, chunks, k = K, weights: list = [0.4, 0.6]):
    """BM25 + Embedding混合检索器 + CrossEncoder重排序"""

    bm25_retriever = BM25Retriever.from_documents(chunks, k=k)
    bm25_retriever.k = k

    vector_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, vector_retriever],
        weights = weights
    )  # EnsembleRetriever内部使用RRF融合策略

    # cross_encoder重排
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    cross_encoder_reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever,
        base_compressor=cross_encoder_reranker,
    )
    
    return compression_retriever

    

