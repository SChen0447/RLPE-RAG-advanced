from app.rag.vectorstore import VectorStore
from app.rag.chains import build_qa_history_multi_query_chain
from app.evaluation.evals import RAGEvaluator
from app.evaluation.dataset_loader import generate_testset
from app.llm import get_llm
from app.rag.embeddings import BGEEmbeddings

from datasets import Dataset

import pandas as pd

testset_path = "data_base/eval/generated_testset.csv"
output_path = "data_base/eval/results/eval_result.csv"

def main():
    vectordb = VectorStore()
    vectordb.load()
    qa_chain = build_qa_history_multi_query_chain(vectordb)

    evaluator = RAGEvaluator(qa_chain, get_llm(), BGEEmbeddings())
    
    df = pd.read_csv(testset_path)
    testset = Dataset.from_pandas(df)

    result = evaluator.run(testset)

    df = result.to_pandas()
    df.to_csv(output_path, index=False)
    print("\n=== 评估结果 ===")
    print(result)
    print(f"\n详细结果已保存到 {output_path}")



if __name__ == "__main__":
    main()