from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.config import MAX_TURNS
from collections import defaultdict

class SessionStore:
    """简易内存会话存储"""
    def __init__(self, max_turns=MAX_TURNS):
        self._store: List[str, List[BaseMessage]] = defaultdict(list)
        self.max_turns = max_turns

    def add_history(self, session_id: str, human: str, ai: str):
        history = self._store[session_id]
        history.append(HumanMessage(content=human))
        history.append(AIMessage(content=ai))
        if len(history) > self.max_turns * 2:
            self._store[session_id] = history[-self.max_turns*2:]

    def get_history(self, session_id):
        return self._store[session_id]


