from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import List
from contextlib import asynccontextmanager

from app.rag.chains import build_qa_history_hyde_chain, build_qa_history_multi_query_chain
from app.rag.vectorstore import VectorStore
from app.session import SessionStore
from app.rag.loaders import get_file_paths, load_documents, split_documents


class AskRequest(BaseModel):
    session_id: str
    question: str

class AskResponse(BaseModel):
    session_id: str
    question: str
    context: List[str]
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    vectordb = VectorStore()
    vectordb.load()

    app.state.vectordb = vectordb
    app.state.chain = build_qa_history_multi_query_chain(vectordb)
    app.state.session = SessionStore()
    yield


app = FastAPI(lifespan= lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> AskResponse:
    chain = app.state.chain
    history = app.state.session.get_history(req.session_id)

    result = chain.invoke({
        "input": req.question,
        "chat_history": history
    })

    answer = result["answer"]
    context = [d.page_content for d in result.get("context", [])]

    app.state.session.add_history(req.session_id, req.question, answer)

    return AskResponse(
        session_id = req.session_id,
        question = req.question,
        context = context,
        answer = answer
        )


# 流式输出
@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    chain = app.state.chain
    history = app.state.session.get_history(req.session_id)

    async def event_gen():
        buffer = []
        async for chunk in chain.astream({
            "input": req.question,
            "chat_history": history,
        }):
            # 只有 qa_chain 里的 StrOutputParser 是流式的，
            # 最后一步 answer 字段是字符串，整体 chain 的输出是字典
            if isinstance(chunk, dict) and "answer" in chunk:
                piece = chunk["answer"]
                buffer.append(piece)
                yield piece
        # 结束后写入历史
        app.state.session.add_history(req.session_id, req.question, "".join(buffer))

    return StreamingResponse(event_gen(), media_type="text/plain; charset=utf-8")
