# vectorizer.py - Converts text into embeddings using Vertex AI

from vertexai.language_models import TextEmbeddingModel


def generate_embedding(text: str):
    embedding_model = TextEmbeddingModel.from_pretrained(
        "textembedding-gecko@latest"
    )
    embedding = embedding_model.get_embeddings([text])[0].values
    return embedding
