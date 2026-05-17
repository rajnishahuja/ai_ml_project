import os
import logging
from typing import Union, Dict, Any
from src.common.schema import RiskReport

logger = logging.getLogger(__name__)

def render_html_report(report: Union[RiskReport, Dict[str, Any]], output_path: str) -> str:
    """
    Renders the RiskReport data structure into a gorgeous pastel-light HTML report.
    
    Args:
        report: Either a RiskReport object or its dictionary representation.
        output_path: Path to write the rendered HTML file.
        
    Returns:
        The generated HTML content as a string.
    """
    try:
        from jinja2 import Template
    except ImportError:
        logger.error("jinja2 package is not installed. Please run 'pip install jinja2'")
        raise

    # Convert RiskReport to dictionary if needed
    if isinstance(report, RiskReport):
        report_data = report.to_dict()
        # Keep low_risk for rendering, since to_dict() might filter it out for standard API outputs
        if hasattr(report, 'low_risk') and report.low_risk:
            # We want the full detailed low_risk list in the HTML view
            report_data['low_risk'] = [
                c.to_dict() if hasattr(c, 'to_dict') else c for c in report.low_risk
            ]
    else:
        report_data = report

    # Load templates and styles
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(base_dir, "templates", "report.html.j2")
    css_path = os.path.join(base_dir, "templates", "styles", "pastel_light.css")

    # Read CSS stylesheet
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_styles = f.read()
    else:
        logger.warning("CSS stylesheet not found at %s. Using fallback empty styles.", css_path)
        css_styles = ""

    # Read HTML Template
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
    else:
        raise FileNotFoundError(f"Jinja2 template not found at {template_path}")

    # Compile Template & Render
    template = Template(template_content)
    rendered_html = template.render(report=report_data, css_styles=css_styles)

    # Save to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    logger.info("Beautiful HTML report successfully saved to %s", output_path)
    return rendered_html
