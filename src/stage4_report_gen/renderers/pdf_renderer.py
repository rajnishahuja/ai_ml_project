import os
import logging
from typing import Union, Dict, Any
from src.common.schema import RiskReport

logger = logging.getLogger(__name__)

def render_pdf_report(
    report: Union[RiskReport, Dict[str, Any]], 
    output_pdf_path: str,
    temp_html_path: str = None
) -> None:
    """
    Compiles and exports the RiskReport to a high-end, print-ready PDF document.
    Utilises WeasyPrint as the primary engine for high-fidelity CSS layout rendering.
    If WeasyPrint is not available, falls back to alternative libraries or provides instructions.
    
    Args:
        report: The RiskReport object or raw report dictionary.
        output_pdf_path: The target file path to save the generated PDF.
        temp_html_path: Optional path for temporary HTML rendering.
    """
    from src.stage4_report_gen.renderers.html_renderer import render_html_report

    # Define temporary html path if not provided
    if not temp_html_path:
        temp_html_path = output_pdf_path.replace(".pdf", "_temp.html")

    # 1. Render the HTML/CSS document first
    logger.info("Generating intermediate HTML representation for PDF compiler...")
    render_html_report(report, temp_html_path)

    # 2. Attempt compilation using WeasyPrint (Gold Standard for HTML to PDF with full CSS support)
    try:
        import weasyprint
        logger.info("Using WeasyPrint engine for high-fidelity PDF compilation...")
        weasyprint.HTML(temp_html_path).write_pdf(output_pdf_path)
        logger.info("High-fidelity PDF report successfully created at %s", output_pdf_path)
        
        # Clean up temporary HTML
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)
        return
    except ImportError:
        logger.info("WeasyPrint is not installed. Trying alternative engines...")

    # 3. Attempt compilation using pdfkit (wkhtmltopdf fallback)
    try:
        import pdfkit
        logger.info("Using pdfkit engine for PDF compilation...")
        pdfkit.from_file(temp_html_path, output_pdf_path)
        logger.info("PDF report successfully created at %s via pdfkit", output_pdf_path)
        
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)
        return
    except ImportError:
        logger.info("pdfkit is not installed.")
    except Exception as e:
        logger.warning("pdfkit compilation failed: %s", str(e))

    # 4. Fallback warning with instructions
    logger.warning(
        "\n" + "="*80 + "\n"
        "PDF COMPILER WARNING:\n"
        "No HTML-to-PDF compilers (weasyprint, pdfkit) were found in the current environment.\n"
        "To enable direct PDF generation, please run:\n"
        "  pip install weasyprint\n\n"
        "For now, the pipeline has generated a beautiful, self-contained HTML report at:\n"
        f"  {temp_html_path}\n"
        "You can open this HTML file in any browser and use 'Print to PDF' (Cmd+P / Ctrl+P)\n"
        "to save it with our premium pastel-light styles perfectly intact!\n"
        + "="*80
    )
