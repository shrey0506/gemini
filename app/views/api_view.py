from fastapi import APIRouter, HTTPException
from app.controllers.chatbot_controller import ChatbotController
from app.controllers.embedding_controller import EmbeddingController

router = APIRouter()
chatbot_controller = ChatbotController()
embedding_controller = EmbeddingController()

@router.post("/chat/")
async def chat(prompt: str):
    try:
        response = chatbot_controller.handle_prompt(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embed/")
async def embed(documents: list[str]):
    try:
        embedding_controller.embed_and_save_documents(documents)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/")
async def search(query: str):
    try:
        results = embedding_controller.search_documents(query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
