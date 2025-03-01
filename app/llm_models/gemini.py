# gemini.py - Gemini Model Integration with Vertex AI SDK

from vertexai.generative_models import GenerativeModel


class GeminiModel:
    def __init__(self, project_id: str, model_version: str):
        self.project_id = project_id
        self.model_version = model_version  # e.g., "gemini-1.5-flash-002"
        self.model = GenerativeModel(self.model_version)

    def call_model(self, prompt: str):
        """Generates a response using the Gemini model."""
        try:
            response = self.model.generate_content(prompt)
            return response.text if response else "No response from Gemini"
        except Exception as e:
            return f"Error in Gemini model: {str(e)}"
