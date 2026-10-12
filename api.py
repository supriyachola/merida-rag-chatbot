
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_chain import ask_question

app = FastAPI(
    title="Merida Tech Minds RAG Chatbot",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return ask_question(request.question)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The chatbot could not process your request."
        )
