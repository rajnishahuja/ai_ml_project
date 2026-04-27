import os
from pathlib import Path

from src.stage1_extract_classify.model import ClauseExtractorClassifier
from src.stage1_extract_classify.preprocessing import preprocess_contract

# Robustly resolve the project root regardless of where the script is executed
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "stage1_2_deberta")

# Path to your fine-tuned model weights (Fallback to the root stage1_2_deberta directory)
MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH)


class Stage1ExtractionService:
    """
    Singleton Wrapper for the Stage 1 CUAD Inference Engine.
    Instantiated once when the FastAPI server boots.
    """

    def __init__(self, model_path: str):
        # Initialize the professional extractor once
        self.extractor = ClauseExtractorClassifier(model_path)

    def infer_from_file(self, file_path: str):
        """
        Processes a file (PDF/DOCX/TXT) from the disk and returns detected legal clauses.
        """
        # 1. Convert file to clean text using the pipeline's helper
        text = preprocess_contract(file_path)
        doc_id = Path(file_path).stem

        # 2. Run the 41-query model logic
        clauses = self.extractor.extract(text, doc_id=doc_id)

        # 3. Convert dataclasses to dicts so FastAPI can send them as JSON
        return [c.to_dict() for c in clauses]

    def infer_from_text(self, text: str, doc_id: str = "custom_text"):
        """
        Processes raw text strings directly (e.g., if uploaded via JSON instead of File).
        """
        clauses = self.extractor.extract(text, doc_id=doc_id)
        return [c.to_dict() for c in clauses]


# Initialize the singleton instance lazily or safely
extraction_service = None


def get_extraction_service() -> Stage1ExtractionService:
    """
    FastAPI Dependency to retrieve the initialized extraction service.
    Ensures model weights are loaded efficiently just once.
    """
    global extraction_service
    if extraction_service is None:
        extraction_service = Stage1ExtractionService(MODEL_PATH)
    return extraction_service
