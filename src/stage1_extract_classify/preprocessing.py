import re
from pathlib import Path

def preprocess_contract(file_path: str) -> str:
    """
    Handles PDF, DOCX, and TXT extraction into plain text for inference.
    """
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
