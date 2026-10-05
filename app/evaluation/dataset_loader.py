from __future__ import annotations

from app.rag.loaders import get_file_paths, load_documents, split_documents

from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig


def generate_testset(
        test_size: int = 20, 
        output_path: str = "data_base/eval/generate_testset.csv",
        llm = None,
        embeddings = None,
    ) -> Dataset:

    file_paths = get_file_paths()
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    
    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # RAGAS 要求 metadata 包含 filename
    for chunk in chunks:
        chunk.metadata["filename"] = chunk.metadata.get("source", "unknown")

    generator = TestsetGenerator(
        llm =wrapped_llm,
        embedding_model =wrapped_embeddings,
    )

    # 并发调用llm
    run_config = RunConfig(
        max_workers=128,      # 并发请求数，根据 API 限流调整
        timeout=180,        # 单个请求超时时间（秒）
        max_retries=3,      # 失败重试次数
        max_wait=60,        # 重试最大等待时间（秒）
    )

    testset = generator.generate_with_langchain_docs(chunks, testset_size=test_size, run_config=run_config)

    df = testset.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"Testset generated to {output_path}, {len(df)} samples") 
    return Dataset.from_pandas(df)