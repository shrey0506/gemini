# llama.py - LLaMA Model Integration with Vertex AI SDK

from vertexai.language_models import TextGenerationModel


class LlamaModel:
    def __init__(self, project_id: str, model_version: str):
        self.project_id = project_id
        self.model_version = model_version  # e.g., "llama-maas"
        self.model = TextGenerationModel.from_pretrained(self.model_version)

    def call_model(self, prompt: str):
        """Generates a response using the LLaMA model."""
        try:
            response = self.model.predict(prompt)
            return response.text if response else "No response from LLaMA"
        except Exception as e:
            return f"Error in LLaMA model: {str(e)}"
