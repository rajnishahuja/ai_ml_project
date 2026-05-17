from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import documents, pipeline

app = FastAPI(
    title="Legal Contract Risk Analyzer API",
    description="End-to-End backend for clause extraction, risk detection, and report generation.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for potential frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registered Active Routers
app.include_router(pipeline.router,       prefix="/api/v1/pipeline", tags=["Full Pipeline"])
app.include_router(documents.router,      prefix="/api/v1/documents", tags=["Database Explorer"])

@app.get("/health", tags=["System"])
async def health_check():
    """Provides a basic health check endpoint for the server."""
    return {"status": "ok", "service": "Legal Contract Risk Analyzer"}

@app.get("/", tags=["System"])
async def root():
    """Root endpoint providing basic API information."""
    return {
        "message": "Welcome to the Legal Contract Risk Analyzer API.",
        "docs": "Navigate to /docs for the Swagger UI."
    }
