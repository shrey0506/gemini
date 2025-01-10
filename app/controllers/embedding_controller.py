import os
from app.models.embedding_model import EmbeddingModel
from app.models.db_model import SessionLocal, DocumentEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # For PDF processing

class EmbeddingController:
    def __init__(self):
        self.embedding_model = EmbeddingModel()

    def embed_documents(self, folder_path="data/"):
        # Read PDF files from the folder and embed their content
        session = SessionLocal()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                with fitz.open(file_path) as pdf:
                    content = ""
                    for page in pdf:
                        content += page.get_text()
                
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    embedding = self.embedding_model.generate_embedding(chunk)
                    doc_embedding = DocumentEmbedding(
                        title=filename,
                        content=chunk,
                        embedding=embedding
                    )
                    session.add(doc_embedding)
        
        session.commit()
        session.close()
