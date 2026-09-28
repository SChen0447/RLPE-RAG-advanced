from __future__ import annotations

from langchain_core._api.beta_decorator import surface_langchain_beta_warnings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from app.llm import get_llm
from typing import List



def build_hyde_transform(llm=None):
    if llm is None:
        llm = get_llm()

    hyde_system_template = (
        "你是一个知识库文档生成助手"
        "请根据用户的问题，写一段可能包含该问题答案的文档段落用于后续检索匹配"
    )
    hyde_system_prompt = ChatPromptTemplate([
        ("system", hyde_system_template),
        ("human", "{query}"),
    ])
    hyde_chain = (
        RunnableLambda(lambda x: {"query": x})
        | hyde_system_prompt 
        | llm 
        | StrOutputParser()
    )
    return hyde_chain
        


def build_multi_query_transform(llm=None, n=2):
    """
    返回一个 Multi-Query 多查询转换 Runnable：
    输入：字符串问题
    输出：查询列表 [原问题, 变体1, ..., 变体n]
    """
    if llm is None:
        llm = get_llm()
    
    multi_query_system_template = (
        "你是一个查询改写助手。请根据用户的问题，生成 {n} 个不同角度的查询，"
        "用于从知识库中检索相关信息。\n"
        "要求：\n"
        "1. 每个查询独立、完整，不要使用代词（如“它”“这个”），不要依赖上下文。\n"
        "2. 从不同侧面、不同关键词组合来提问，覆盖可能的相关信息。\n"
        "3. 每行输出一个查询，不要编号，不要输出其他任何内容。\n"
    )

    multi_query_system_prompt = ChatPromptTemplate([
        ("system", multi_query_system_template),
        ("human", "{query}"),
    ])

    multi_query_chain = (
        RunnableLambda(lambda x: {"query": x, "n": n})
        | multi_query_system_prompt 
        | llm 
        | StrOutputParser()
    )

    def parse_multi_query_output(origin_question: str) -> List[str]:
        sub_questions_str = multi_query_chain.invoke(origin_question)
        sub_question_list = [q.strip() for q in sub_questions_str.strip().split("\n")]
        return [origin_question] + sub_question_list


    return RunnableLambda(parse_multi_query_output)






