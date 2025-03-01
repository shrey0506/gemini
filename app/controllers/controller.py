# controller.py - Handles model selection and LLM call

from app.config.config_loader import load_config

from app.llm_models.gemini import GeminiModel
from app.llm_models.llama import LlamaModel

config = load_config()


def call_llm(conversation_id, user_id, question, model):
    """Handles LLM model selection and API call."""
    project_id = config["models"][model.name]["project_id"]

    if model.name == "gemini":
        llm = GeminiModel(project_id, model.version)
    elif model.name == "llama":
        llm = LlamaModel(project_id, model.version)
    else:
        return {"error": "Invalid model name"}

    response = llm.call_model(question)
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "response": response,
    }
