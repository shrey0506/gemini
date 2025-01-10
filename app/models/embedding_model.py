import os
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class EmbeddingModel:
    def __init__(self, embeddings_dir="embeddings"):
        self.embeddings_dir = embeddings_dir
        os.makedirs(embeddings_dir, exist_ok=True)

    def embed_documents(self, documents):
        embeddings = OpenAIEmbeddings()
        for doc_id, content in enumerate(documents):
            vector = embeddings.embed(content)
            with open(os.path.join(self.embeddings_dir, f"doc_{doc_id}.json"), "w") as f:
                json.dump({"id": doc_id, "vector": vector}, f)

    def search(self, query, top_k=5):
        embeddings = OpenAIEmbeddings()
        query_vector = embeddings.embed(query)

        similarities = []
        for file in os.listdir(self.embeddings_dir):
            with open(os.path.join(self.embeddings_dir, file), "r") as f:
                data = json.load(f)
                similarity = cosine_similarity([query_vector], [data["vector"]])[0][0]
                similarities.append((data["id"], similarity))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
