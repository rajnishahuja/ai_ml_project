import re
from pathlib import Path


def preprocess_contract(file_path: str) -> str:
    """Handles PDF, DOCX, and TXT extraction."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext in (".docx", ".doc"):
        from docx import Document

        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif ext == ".txt":
        text = Path(file_path).read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {ext}")

    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
