import sys
from app.controllers.embedding_controller import EmbeddingController
from app.controllers.chatbot_controller import ChatbotController

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "embed":
        embedding_controller = EmbeddingController()
        embedding_controller.embed_documents()
        print("Document embedding process completed!")
    else:
        print("Usage: python main.py embed")
