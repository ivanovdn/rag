from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes.query import router

app = FastAPI(
    title="Compliance Bot API",
    description="Internal compliance policy Q&A",
    version="1.0.0",
)

# CORS — allow Teams bot and local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": settings.pipeline_mode,
        "llm": settings.llm_model,
    }
