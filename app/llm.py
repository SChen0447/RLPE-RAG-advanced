from langchain_openai import ChatOpenAI
from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TEMPERATURE


def get_llm():
    """获取DeepSeek LLM"""
    llm = ChatOpenAI(
        model_name=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=TEMPERATURE
    )
    
    return llm




