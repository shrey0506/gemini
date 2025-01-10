from fastapi import FastAPI
from app.views.api_view import router

app = FastAPI()

# Register the API router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
