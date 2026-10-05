from __future__ import annotations

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from typing import List
from langchain_core.documents import Document

from ragas.metrics import (
    faithfulness,           # 忠实度：答案是否基于检索到的上下文
    answer_relevancy,       # 答案相关性：答案是否切题
    context_precision,      # 上下文精准度：检索到的上下文是否与问题相关
    context_recall,         # 上下文召回率：检索到的上下文是否与问题相关
    answer_correctness,     # 答案准确性：与ground_truth的匹配度
)


class RAGEvaluator:
    def __init__(self, qa_chain, llm , embeddings):
        self.qa_chain = qa_chain
        self.llm = LangchainLLMWrapper(llm)
        self.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def _run_chain(self, question: str) -> dict:
        """跑一次 RAG 链，收集 answer 和 contexts。"""
        result = self.qa_chain.invoke({
            "input": question,
            "chat_history": [],
        })

        # 兼容两种输出：字典或字符串
        if isinstance(result, dict):
            answer = result.get("answer", "")
            contexts = result.get("context", [])
        else:
            answer = str(result)
            contexts = []

        # contexts 统一成 list[str]
        if isinstance(contexts, str):
            contexts = [contexts]
        elif not isinstance(contexts, list):
            contexts = []
        elif isinstance(contexts, list) and contexts and isinstance(contexts[0], Document):
            contexts = [str(x.page_content) for x in contexts]

        return {"answer": answer, "contexts": contexts}



    def run(self, dataset: Dataset) -> "EvaluationResult":
        questions = dataset["user_input"]
        ground_truths = dataset["reference"]

        answers = []
        contexts = []

        for q in questions:
            output = self._run_chain(q)
            answers.append(output["answer"])
            contexts.append(output["contexts"])

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": contexts,
        })

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]

        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        return result
        
