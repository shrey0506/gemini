# agent.py - Implements Agentic Flow with Vector Embeddings & AlloyDB Search

from app.controllers.controller import call_llm
from app.db.alloydb import search_similar_documents
from app.utils.vectorizer import generate_embedding


class Agent:
    def __init__(self, model_name="gemini"):
        self.model_name = model_name

    def execute(self, conversation_id, user_id, question):
        # Step 1: Convert query to embedding
        query_embedding = generate_embedding(question)

        # Step 2: Search AlloyDB for relevant documents
        retrieved_docs = search_similar_documents(query_embedding)

        # Step 3: Combine query with relevant context and call LLM
        context = (
            "\n".join(retrieved_docs)
            if retrieved_docs
            else "No relevant documents found."
        )
        enhanced_prompt = f"Context: {context}\n\nUser Query: {question}"

        # Step 4: Get response from LLM
        response = call_llm(
            conversation_id,
            user_id,
            enhanced_prompt,
            self.model_name
        )

        return {"task": "Retrieval-Augmented Response", "response": response}
