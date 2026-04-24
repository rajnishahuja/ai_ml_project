import re
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_contract_pdfplumber(file_path: str, doc_id: str = None) -> str:
    """Original fast, brute-force extraction using pdfplumber."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                if page_text := page.extract_text(): 
                    text_parts.append(page_text)
        text = "\n\n".join(text_parts)
    elif ext in (".docx", ".doc"):
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".txt":
        text = Path(file_path).read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    text = re.sub(r" {2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def preprocess_contract_docling(file_path: str, doc_id: str = None) -> str:
    """
    Advanced layout-aware extraction using IBM Docling.
    Saves raw JSON metadata to disk for Stage 4 tracking, 
    and returns perfectly stitched Markdown text.
    """
    ext = Path(file_path).suffix.lower()
    # Docling handles PDF, DOCX natively
    if ext == ".txt":
        return preprocess_contract_pdfplumber(file_path, doc_id)
        
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        logger.error("Docling not installed. Falling back to pdfplumber.")
        return preprocess_contract_pdfplumber(file_path, doc_id)
        
    logger.info(f"Running layout-aware Docling parser on {Path(file_path).name}...")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc_dict = result.document.export_to_dict()
    
    # Save the raw JSON to disk for manual review and later Stage 4 use
    base_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = base_dir / "data" / "processed" / "docling_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{doc_id}.json" if doc_id else f"{Path(file_path).stem}.json"
    out_file = out_dir / filename
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2)
    logger.info(f"Saved raw Docling JSON to {out_file}")
    
    # Export perfectly stitched Markdown for DeBERTa
    text = result.document.export_to_markdown()
    
    text = re.sub(r" {2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def preprocess_contract(file_path: str, doc_id: str = None) -> str:
    """
    Main entry point. Uses USE_PDFPLUMBER environment variable to toggle engine.
    Default is Docling (advanced layout-aware extraction) for optimal accuracy.
    """
    use_pdfplumber = os.getenv("USE_PDFPLUMBER", "false").lower() == "true"
    
    if use_pdfplumber:
        return preprocess_contract_pdfplumber(file_path, doc_id)
    else:
        return preprocess_contract_docling(file_path, doc_id)
