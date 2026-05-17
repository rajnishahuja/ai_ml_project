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
    Saves raw JSON metadata and stitched Markdown cache to disk,
    allowing instant resume if the pipeline is interrupted.
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
        
    # Set up Cache paths directly in the contract's final output folder
    base_dir = Path(__file__).resolve().parent.parent.parent
    clean_doc_id = doc_id if doc_id else Path(file_path).stem
    out_dir = base_dir / "data" / "output" / "final" / clean_doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file_json = out_dir / "stage1_output.json"
    out_file_md = out_dir / "stage1_layout.txt"
    
    # Layer 2 Cache Hit Check: Skip conversion if layout caches already exist!
    if out_file_json.exists() and out_file_md.exists():
        logger.info(f"Layout Cache Hit: Found existing Docling markdown cache at {out_file_md}. Skipping PDF converter conversion!")
        with open(out_file_md, "r", encoding="utf-8") as f:
            text = f.read()
        text = re.sub(r" {2,}", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()
        
    logger.info(f"Running layout-aware Docling parser on {Path(file_path).name}...")
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc_dict = result.document.export_to_dict()
    
    # Save the raw JSON to disk for Stage 4 use
    with open(out_file_json, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2)
    logger.info(f"Saved raw Docling JSON to {out_file_json}")
    
    # Export perfectly stitched Markdown for DeBERTa
    text = result.document.export_to_markdown()
    
    # Save the stitched Markdown to disk cache to skip future layout parsing
    with open(out_file_md, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved stitched Docling Markdown cache to {out_file_md}")
    
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
