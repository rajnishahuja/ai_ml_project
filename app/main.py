from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import stage1_extract, documents
# from app.routers import stage3_agent, stage4_report

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
app.include_router(stage1_extract.router, prefix="/api/v1/stage1", tags=["Extraction & Classification"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Database Explorer"])
# app.include_router(stage3_agent.router, prefix="/api/v1/stage3", tags=["Risk Detection Agent"])
# app.include_router(stage4_report.router, prefix="/api/v1/stage4", tags=["Report Generation"])

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
