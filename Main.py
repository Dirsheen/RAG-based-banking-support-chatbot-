"""
Banking Support Chatbot - FastAPI Backend
Entry point: registers routes, CORS, startup events.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers import chat, upload, health
from services.ingestion import seed_sample_documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: seed sample banking docs if DB is empty."""
    print("Starting Banking Support Chatbot API...")
    await seed_sample_documents()
    print("Vector DB ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Banking Support Chatbot API",
    description="RAG-powered chatbot for banking customer support",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin (update in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Documents"])
app.include_router(chat.router, tags=["Chat"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )
