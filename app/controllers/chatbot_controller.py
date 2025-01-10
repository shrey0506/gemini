from app.models.embedding_model import EmbeddingModel
from app.models.llm_model import LLMModel
from app.models.db_model import SessionLocal, DocumentEmbedding

class ChatbotController:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.llm_model = LLMModel()

    def handle_prompt(self, prompt):
        # Embed the user's query
        query_embedding = self.embedding_model.generate_embedding(prompt)

        # Retrieve similar embeddings from AlloyDB
        session = SessionLocal()
        documents = session.query(DocumentEmbedding).all()
        results = [
            (doc, self.embedding_model.calculate_similarity(query_embedding, doc.embedding))
            for doc in documents
        ]
        results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

        # Pass relevant context to the LLM
        context = "
".join([doc.content for doc, _ in results])
        response = self.llm_model.generate_response(f"{context}

User Query: {prompt}")
        session.close()

        return response
