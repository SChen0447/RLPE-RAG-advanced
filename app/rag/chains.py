from __future__ import annotations
from typing import List
from app.rag.vectorstore import VectorStore
from app.rag.retriever import build_hybrid_retriever, rerank_cross_encoder, build_raw_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from app.llm import get_llm
from app.rag.query_transform import build_hyde_transform, build_multi_query_transform
from app.config import K

import hashlib


def format_docs(docs: dict):
    """把检索到的 Document 列表拼成上下文字符串。"""
    if not docs:
        return "（没有检索到相关资料）"
    return "\n\n".join(doc.page_content for doc in docs["context"])


# 历史信息压缩+hyde
def build_qa_history_hyde_chain(vectordb: VectorStore):
    retriever = build_hybrid_retriever(vectordb.get_vectorstore())
    llm = get_llm()

    hyde_chain = build_hyde_transform(llm)

    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题,如果没有多余的聊天记录则返回用户的问题"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    retrieval_chain = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | hyde_chain | retriever),
        condense_question_prompt | llm | StrOutputParser() | hyde_chain | retriever
    )



    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieval_chain,
    ).assign(answer=qa_chain)

    return qa_history_chain


# 历史信息压缩+多查询
def build_qa_history_multi_query_chain(vectordb: VectorStore):
    retriever = build_raw_retriever(vectordb.get_vectorstore())
    llm = get_llm()
    multi_query_chain = build_multi_query_transform(llm)
    
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题,如果没有多余的聊天记录则返回用户的问题"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    def retriever_and_merge(queries: list[str], top_k: int = K) -> List[Document]:
        seen = set()
        merged = []
        for query in queries:
            for doc in retriever.invoke(query):
                key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        merged = rerank_cross_encoder(queries[0], merged, top_k)
        return merged

    retrieval_chain = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | multi_query_chain | RunnableLambda(retriever_and_merge)),
        condense_question_prompt | llm | StrOutputParser() | multi_query_chain | RunnableLambda(retriever_and_merge)
    )


    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieval_chain,
    ).assign(answer=qa_chain)

    return qa_history_chain












