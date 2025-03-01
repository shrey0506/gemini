import os
import psycopg2

DB_CONFIG = {
    "dbname": os.getenv("ALLOYDB_DBNAME", "your_db"),
    "user": os.getenv("ALLOYDB_USER", "your_user"),
    "password": os.getenv("ALLOYDB_PASSWORD", "your_password"),
    "host": os.getenv("ALLOYDB_HOST", "your_host"),
    "port": os.getenv("ALLOYDB_PORT", "5432"),
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def save_document(conversation_id, user_id, document_text, vector_embedding):
    """Saves document and its embedding to AlloyDB."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (conversation_id, user_id, content, embedding) VALUES (%s, %s, %s, %s)",
        (conversation_id, user_id, document_text, vector_embedding)
    )
    conn.commit()
    cur.close()
    conn.close()

def search_similar_documents(query_embedding, top_k=5):
    """Searches for top-k similar documents using vector similarity."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT content FROM documents ORDER BY embedding <-> %s LIMIT %s",
        (query_embedding, top_k)
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]
