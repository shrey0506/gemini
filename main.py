# main.py - Entry point for FastAPI application

from fastapi import FastAPI
from app.views.view import router as api_router

app = FastAPI(
    title="Agentic AI Solution",
    description="FastAPI-based Agentic AI solution with LLMs and AlloyDB integration.",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Agentic AI Solution is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
