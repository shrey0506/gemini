# view.py - FastAPI endpoints

from fastapi import APIRouter
from pydantic import BaseModel

from app.controllers.controller import handle_chat, handle_upload

router = APIRouter()


# Pydantic model for request validation
class ModelConfig(BaseModel):
    name: str  # "gemini" or "llama"
    version: str  # e.g., "gemini-1.5-flash-002" or "llama-maas"


@router.post("/chat")
async def chat_api(
    conversation_id: str, user_id: str, question: str, model: ModelConfig
):
    return handle_chat(conversation_id, user_id, question, model)


@router.post("/upload")
async def upload_api(
    gcs_signed_url: str, conversation_id: str, user_id: str, document_text: str
):
    return handle_upload(
        gcs_signed_url,
        conversation_id,
        user_id,
        document_text
    )
