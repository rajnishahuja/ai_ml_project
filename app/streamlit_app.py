import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Insert project root into system path to allow local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage4_report_gen.report_builder import export_markdown_report

# Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="Legal Contract Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS Styling Injection for a gorgeous enterprise theme
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Core Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .banner-title {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek Title Banner */
    .banner-title {
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    
    .banner-subtitle {
        font-size: 1.15rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 28px;
    }
    
    /* Premium Metric Card Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -4px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    /* Custom Clause Card Callouts */
    .clause-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    
    .clause-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Interactive Pills */
    .badge {
        padding: 5px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-medium { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .badge-low { background-color: #D1FAE5; color: #065F46; border: 1px solid #6EE7B7; }
    
    /* Styled Progress Steps */
    .step-row {
        display: flex;
        align-items: center;
        width: 100%;
        box-sizing: border-box;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 8px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        font-size: 0.95rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinner-icon {
        border: 2px solid #E2E8F0;
        border-top: 2px solid #2563EB;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    @keyframes pulse-border {
        0%, 100% { 
            border-color: #CBD5E1; 
            box-shadow: 0 0 0 0px rgba(59, 130, 246, 0);
        }
        50% { 
            border-color: #3B82F6; 
            box-shadow: 0 0 10px 2px rgba(59, 130, 246, 0.15);
        }
    }
    
    @keyframes pulse-details {
        0%, 100% { opacity: 0.55; color: #475569; }
        50% { opacity: 1.0; color: #2563EB; }
    }
    
    .step-active {
        animation: pulse-border 2s infinite ease-in-out;
        background-color: #F8FAFC !important;
    }
    
    .step-active i {
        animation: pulse-details 1.5s infinite ease-in-out;
        font-weight: 500;
    }
    
    /* Vault Grid Custom Cards */
    .vault-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 20px;
    }
    
    .vault-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px -8px rgba(37, 99, 235, 0.15);
        border-color: #2563EB;
    }
    
    .vault-card-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    
    .vault-card-date {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 16px;
    }
    
    .vault-stats-row {
        display: flex;
        gap: 16px;
        margin-bottom: 0px;
        font-size: 0.9rem;
    }
    
    /* Enlarge Streamlit Tab Labels */
    button[data-baseweb="tab"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
    }
    
    /* Large telemetry badges styling */
    .telemetry-badge {
        font-size: 1.05rem !important;
        font-weight: 700;
        color: #047857;
        background-color: #ECFDF5;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #A7F3D0;
        display: inline-block;
    }
    .telemetry-badge-blue {
        font-size: 1.05rem !important;
        font-weight: 700;
        color: #0369A1;
        background-color: #F0F9FF;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #BAE6FD;
        display: inline-block;
    }
    .telemetry-badge-purple {
        font-size: 1.05rem !important;
        font-weight: 700;
        color: #6B21A8;
        background-color: #FAF5FF;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #F3E8FF;
        display: inline-block;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# Synchronize Streamlit Query Parameters with Session State for web-native card links
query_params = st.query_params
if "doc_id" in query_params:
    st.session_state.selected_doc = query_params["doc_id"]
else:
    st.session_state.selected_doc = None


# Helper to format timestamps beautifully
def format_datetime_beautiful(dt_str: str) -> str:
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("🗓️ %A, %B %d, %Y at %I:%M %p (UTC)")
    except Exception:
        return f"🗓️ {dt_str}"


# Helpers to find completed and interrupted analyses
def get_completed_analyses_metadata():
    output_dir = PROJECT_ROOT / "data" / "output" / "final"
    if not output_dir.exists():
        return []
    completed = []
    for f in output_dir.iterdir():
        if f.is_dir() and (f / "final_report.json").exists():
            try:
                with open(f / "final_report.json", "r", encoding="utf-8") as file:
                    data = json.load(file)
                mtime = f.stat().st_mtime
                completed.append(
                    {
                        "id": f.name,
                        "score": data.get("overall_risk_score", 0.0),
                        "total_clauses": data.get("total_clauses", 0),
                        "missing": len(data.get("missing_protections", [])),
                        "mtime": mtime,
                    }
                )
            except Exception:
                pass
    completed.sort(key=lambda x: x["mtime"], reverse=True)
    return completed


def get_interrupted_analyses():
    output_dir = PROJECT_ROOT / "data" / "output" / "final"
    
    if not output_dir.exists():
        return []

    interrupted = []
    for doc_dir in output_dir.iterdir():
        if doc_dir.is_dir():
            doc_id = doc_dir.name
            has_final = (doc_dir / "final_report.json").exists()
            has_stage1 = (doc_dir / "stage1_output.json").exists()
            has_stage2 = (doc_dir / "stage2_output.json").exists()
            
            # If the audit started (either Docling layout parsed or DeBERTa extracted) but is not complete
            if (has_stage1 or has_stage2) and not has_final:
                interrupted.append(doc_id)
            
    return interrupted


# 100% Horizontal Sidebarless Layout Branding


# Execution Stepped Pipeline Logic
def execute_stepped_pipeline(pdf_path, doc_id):
    import requests
    import time
    import concurrent.futures

    # UI stepped updates container
    step_container = st.container()
    with step_container:
        st.markdown("### ⚙️ Compliance Audit Progress Timeline")

        step1 = st.empty()
        step2 = st.empty()
        step3 = st.empty()
        step4 = st.empty()

        # Paths to monitor on disk
        final_dir = PROJECT_ROOT / "data" / "output" / "final" / doc_id
        stage1_path = final_dir / "stage1_output.json"
        stage2_path = final_dir / "stage2_output.json"
        stage3_path = final_dir / "stage3_output.json"
        final_report_path = final_dir / "final_report.json"

        # Dynamic stepped timeline renderer with stage-aware failure rendering
        def render_timeline(s1_state, s2_state, s3_state, s4_state, error_msg=None):
            # Stage 1: Document Structure & Layout Parsing
            if s1_state == "running":
                step1.markdown(
                    "<div class='step-row step-active'><div class='spinner-icon'></div><b>Stage 1: Document Structure & Layout Parsing</b> &nbsp;&mdash;&nbsp; <i>Extracting layout segments and reading PDF pages...</i></div>",
                    unsafe_allow_html=True,
                )
            elif s1_state == "done":
                step1.markdown(
                    "<div class='step-row' style='background-color:#F0FDF4; border-color:#DCFCE7;'>✅ <b>Stage 1: Document Layout Successfully Parsed</b></div>",
                    unsafe_allow_html=True,
                )
            elif s1_state == "failed":
                step1.markdown(
                    f"<div class='step-row' style='background-color:#FEF2F2; border-color:#FEE2E2; color:#991B1B;'>❌ <b>Stage 1 Failed: Document Layout Parsing Interrupted</b><br><i style='font-size: 0.9em; margin-left: 20px; display: block; margin-top: 5px;'>Error: {error_msg}</i></div>",
                    unsafe_allow_html=True,
                )
            else:
                step1.markdown(
                    "<div class='step-row' style='background-color:#F9FAFB; border-color:#E5E7EB; color:#9CA3AF;'>⏳ <b>Stage 1: Document Structure & Layout Parsing</b> &nbsp;&mdash;&nbsp; <i>Pending...</i></div>",
                    unsafe_allow_html=True,
                )

            # Stage 2: Clause Span Extraction
            if s2_state == "running":
                step2.markdown(
                    "<div class='step-row step-active'><div class='spinner-icon'></div><b>Stage 2: Clause Span Extraction</b> &nbsp;&mdash;&nbsp; <i>Parsing document text into distinct semantic clauses...</i></div>",
                    unsafe_allow_html=True,
                )
            elif s2_state == "done":
                step2.markdown(
                    "<div class='step-row' style='background-color:#F0FDF4; border-color:#DCFCE7;'>✅ <b>Stage 2: Clause Span Extraction Complete</b></div>",
                    unsafe_allow_html=True,
                )
            elif s2_state == "failed":
                step2.markdown(
                    f"<div class='step-row' style='background-color:#FEF2F2; border-color:#FEE2E2; color:#991B1B;'>❌ <b>Stage 2 Failed: Clause Span Extraction Interrupted</b><br><i style='font-size: 0.9em; margin-left: 20px; display: block; margin-top: 5px;'>Error: {error_msg}</i></div>",
                    unsafe_allow_html=True,
                )
            else:
                step2.markdown(
                    "<div class='step-row' style='background-color:#F9FAFB; border-color:#E5E7EB; color:#9CA3AF;'>⏳ <b>Stage 2: Clause Span Extraction</b> &nbsp;&mdash;&nbsp; <i>Pending...</i></div>",
                    unsafe_allow_html=True,
                )

            # Stage 3: Risk Classification & Precedent Audit
            if s3_state == "running":
                ckpt_path = final_dir / "stage3_checkpoint.jsonl"
                audited_count = 0
                if ckpt_path.exists():
                    try:
                        with open(ckpt_path, "r", encoding="utf-8") as f_ckpt:
                            audited_count = sum(1 for line in f_ckpt if line.strip())
                    except Exception:
                        pass
                
                desc = "Classifying triggers and running legal verifications..."
                if audited_count > 0:
                    desc = f"Audited {audited_count} clauses via LangGraph verification agents..."

                step3.markdown(
                    f"<div class='step-row step-active'><div class='spinner-icon'></div><b>Stage 3: Risk Classification & Agentic Precedent Audit</b> &nbsp;&mdash;&nbsp; <i>{desc}</i></div>",
                    unsafe_allow_html=True,
                )
            elif s3_state == "done":
                step3.markdown(
                    "<div class='step-row' style='background-color:#F0FDF4; border-color:#DCFCE7;'>✅ <b>Stage 3: Risk Classification & Precedent Audit Complete</b> &nbsp;&mdash;&nbsp; <i>Evaluated all clauses using precedent RAG & verifications.</i></div>",
                    unsafe_allow_html=True,
                )
            elif s3_state == "failed":
                step3.markdown(
                    f"<div class='step-row' style='background-color:#FEF2F2; border-color:#FEE2E2; color:#991B1B;'>❌ <b>Stage 3 Failed: Agentic Precedent Audit Interrupted</b><br><i style='font-size: 0.9em; margin-left: 20px; display: block; margin-top: 5px;'>Error: {error_msg}</i></div>",
                    unsafe_allow_html=True,
                )
            else:
                step3.markdown(
                    "<div class='step-row' style='background-color:#F9FAFB; border-color:#E5E7EB; color:#9CA3AF;'>⏳ <b>Stage 3: Risk Classification & Agentic Precedent Audit</b> &nbsp;&mdash;&nbsp; <i>Pending...</i></div>",
                    unsafe_allow_html=True,
                )

            # Stage 4: Executive Synthesis & Report Generation
            if s4_state == "running":
                step4.markdown(
                    "<div class='step-row step-active'><div class='spinner-icon'></div><b>Stage 4: Executive Synthesis & Report Generation</b> &nbsp;&mdash;&nbsp; <i>Synthesizing compliance guidelines and writing final audit report...</i></div>",
                    unsafe_allow_html=True,
                )
            elif s4_state == "done":
                step4.markdown(
                    "<div class='step-row' style='background-color:#F0FDF4; border-color:#DCFCE7;'>✅ <b>Stage 4 Complete: Compliance report successfully generated!</b></div>",
                    unsafe_allow_html=True,
                )
            elif s4_state == "failed":
                step4.markdown(
                    f"<div class='step-row' style='background-color:#FEF2F2; border-color:#FEE2E2; color:#991B1B;'>❌ <b>Stage 4 Failed: Executive Synthesis & Report Generation Interrupted</b><br><i style='font-size: 0.9em; margin-left: 20px; display: block; margin-top: 5px;'>Error: {error_msg}</i></div>",
                    unsafe_allow_html=True,
                )
            else:
                step4.markdown(
                    "<div class='step-row' style='background-color:#F9FAFB; border-color:#E5E7EB; color:#9CA3AF;'>⏳ <b>Stage 4: Executive Synthesis & Report Generation</b> &nbsp;&mdash;&nbsp; <i>Pending...</i></div>",
                    unsafe_allow_html=True,
                )

        # Initial Render
        render_timeline("running", "pending", "pending", "pending")

        # Submit background thread to execute the API `/analyze` request
        api_url = "http://127.0.0.1:8000/api/v1/pipeline/analyze"
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def run_api_call():
            import io
            if pdf_path.exists():
                with open(pdf_path, "rb") as f_pdf:
                    files = {"file": (pdf_path.name, f_pdf, "application/pdf")}
                    return requests.post(api_url, files=files, timeout=600)
            else:
                # Fallback: Upload a dummy empty stream to satisfy API validation.
                # Backend cache check will immediately hit the layout cache on disk and ignore this file!
                dummy_file = io.BytesIO(b"")
                files = {"file": (f"{doc_id}.pdf", dummy_file, "application/pdf")}
                return requests.post(api_url, files=files, timeout=600)

        api_future = executor.submit(run_api_call)

        # Polling Loop to update progressive stepped timeline steps dynamically
        while not api_future.done():
            # Update states based on disk file presence
            if final_report_path.exists():
                render_timeline("done", "done", "done", "running")
            elif stage3_path.exists():
                render_timeline("done", "done", "done", "running")
            elif stage2_path.exists():
                render_timeline("done", "done", "running", "pending")
            elif stage1_path.exists():
                render_timeline("done", "running", "pending", "pending")
            else:
                # Stage 1 (Docling Layout Parsing) is running initially
                render_timeline("running", "pending", "pending", "pending")
            
            time.sleep(1.5)

        # Retrieve background execution results
        try:
            response = api_future.result()
            if response.status_code == 200:
                render_timeline("done", "done", "done", "done")
                st.balloons()
                time.sleep(1.5)
            else:
                try:
                    res_json = response.json()
                    error_detail = res_json.get("detail", response.text)
                    if isinstance(error_detail, list):
                        error_detail = ", ".join([f"{err.get('msg', 'Validation error')} ({'.'.join(str(x) for x in err.get('loc', []))})" for err in error_detail])
                except Exception:
                    error_detail = f"Internal Server Error (Status {response.status_code})"

                # Diagnose failed stage based on checkpoint file presence on disk
                if stage3_path.exists():
                    # Stage 3 completed, meaning Stage 4 failed
                    render_timeline("done", "done", "done", "failed", error_msg=error_detail)
                elif stage2_path.exists():
                    # Stage 2 completed, meaning Stage 3 failed
                    render_timeline("done", "done", "failed", "pending", error_msg=error_detail)
                elif stage1_path.exists():
                    # Stage 1 completed, meaning Stage 2 failed
                    render_timeline("done", "failed", "pending", "pending", error_msg=error_detail)
                else:
                    # No intermediate checkpoint, meaning Stage 1 failed
                    render_timeline("failed", "pending", "pending", "pending", error_msg=error_detail)
                st.stop()

        except Exception as e:
            # Diagnose failed stage for connection / timeout exceptions
            import requests.exceptions
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout)):
                error_msg = "The Legal Risk Analyzer backend server is not running on port 8000. Please start the server by running 'uv run uvicorn app.main:app --port 8000' in a new terminal."
            else:
                error_msg = str(e)
                
            if stage3_path.exists():
                render_timeline("done", "done", "done", "failed", error_msg=error_msg)
            elif stage2_path.exists():
                render_timeline("done", "done", "failed", "pending", error_msg=error_msg)
            elif stage1_path.exists():
                render_timeline("done", "failed", "pending", "pending", error_msg=error_msg)
            else:
                render_timeline("failed", "pending", "pending", "pending", error_msg=error_msg)
            st.stop()


# ---------------------------------------------------------
# Dynamic Selection Flow State
# ---------------------------------------------------------
selected_doc = st.session_state.selected_doc

# ---------------------------------------------------------
# Main Page Dashboard Flow
# ---------------------------------------------------------
if selected_doc is None:
    # Intercept dashboard rendering if an active audit is running in the background!
    active_audit = st.session_state.get("active_audit")
    if active_audit:
        raw_path = PROJECT_ROOT / "data" / "raw" / f"{active_audit}.pdf"
        if raw_path.exists():
            execute_stepped_pipeline(raw_path, active_audit)
            st.query_params["doc_id"] = active_audit
            st.session_state.selected_doc = active_audit
            st.rerun()

    # HOMEPAGE VAULT VIEW
    st.markdown(
        "<div class='banner-title'>⚖️ Legal Contract Risk Analyzer</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='banner-subtitle'>AI-Powered Clause Extraction, Risk Profiling & Precedent Auditing</div>",
        unsafe_allow_html=True,
    )

    # Horizontal Audit Checkpoint Recovery (Always visible!)
    st.markdown("## 🔄 Audit Checkpoint Recovery")
    interrupted = get_interrupted_analyses()
    if interrupted:
        st.warning(
            "⚠️ **Suspended Compliance Audits Detected**: The checkpointer recovered previous contract reviews that were suspended. Resume them immediately to finish:"
        )
        r_col1, r_col2 = st.columns([3, 1])
        with r_col1:
            resume_doc = st.selectbox(
                "Select interrupted audit to resume:",
                interrupted,
                label_visibility="collapsed",
            )
        with r_col2:
            if st.button("🔄 Resume Audit Now", use_container_width=True):
                st.session_state["_dashboard_resume"] = resume_doc

        # Execute OUTSIDE column context so timeline renders full-width
        if st.session_state.get("_dashboard_resume"):
            resume_doc = st.session_state.pop("_dashboard_resume")
            raw_path = PROJECT_ROOT / "data" / "raw" / (resume_doc + ".pdf")
            stage1_cache = PROJECT_ROOT / "data" / "output" / "final" / resume_doc / "stage1_output.json"
            st.query_params["doc_id"] = resume_doc
            st.session_state.selected_doc = resume_doc
            
            if not raw_path.exists() and not stage1_cache.exists():
                st.error(
                    f"Raw source PDF not found at {raw_path}. Please re-upload the contract."
                )
            else:
                execute_stepped_pipeline(raw_path, resume_doc)
                st.rerun()
    else:
        st.info(
            "🎉 **All Systems Active**: No suspended or interrupted audits detected in the database. Your vault is fully synchronized."
        )

    st.write("")

    # Primary CTA: Upload contract centered prominently
    st.markdown("## 📂 Audit New Contract")

    if "uploaded_file_data" not in st.session_state:
        st.session_state.uploaded_file_data = None
        st.session_state.uploaded_file_name = ""

    if st.session_state.uploaded_file_data is None:
        st.markdown(
            """
        <div style="background-color: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px;">
            <p style="font-size: 1.15rem; font-weight: 600; color: #475569; margin-bottom: 8px;">Upload Legal PDF Contract File</p>
            <p style="font-size: 0.9rem; color: #94A3B8; margin-top:-10px;">Drag and drop or select file for 4-Stage deep ML analysis</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Select contract file to audit:",
            type=["pdf"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            key="contract_pdf_uploader",
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file_data = uploaded_file.getvalue()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.rerun()
    else:
        st.markdown(
            f"""
        <div style="background-color: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 12px; padding: 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 0.85rem; color: #0369A1; font-weight: 700; margin: 0; text-transform: uppercase; letter-spacing: 0.05em;">Selected Legal Contract</p>
                    <p style="font-size: 1.2rem; font-weight: 800; color: #0C4A6E; margin: 4px 0 0 0;">📄 {st.session_state.uploaded_file_name}</p>
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #059669; background-color: #D1FAE5; padding: 6px 12px; border-radius: 9999px; border: 1px solid #A7F3D0;">
                    ✓ Ready to Audit
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        act_col1, act_col2 = st.columns([4, 1])
        with act_col1:
            if st.button(
                "⚡ Initiate Stepped Compliance Audit", use_container_width=True
            ):
                doc_id = Path(st.session_state.uploaded_file_name).stem
                raw_dir = PROJECT_ROOT / "data" / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                temp_path = raw_dir / st.session_state.uploaded_file_name

                with open(temp_path, "wb") as f:
                    f.write(st.session_state.uploaded_file_data)

                st.session_state.uploaded_file_data = None
                st.session_state.uploaded_file_name = ""

                execute_stepped_pipeline(temp_path, doc_id)
                st.query_params["doc_id"] = doc_id
                st.session_state.selected_doc = doc_id
                st.rerun()

        with act_col2:
            if st.button("❌ Change File", use_container_width=True):
                st.session_state.uploaded_file_data = None
                st.session_state.uploaded_file_name = ""
                st.rerun()

    st.markdown("---")

    # 2. Audited Contracts Vault (Clickable cards wrapped fully in HTML anchor links!)
    st.markdown("## 🗂️ Audited Contracts Vault")
    completed = get_completed_analyses_metadata()

    if not completed:
        st.info(
            "No audited contracts found in archive. Upload a PDF contract above to run your very first Deep Risk Audit!"
        )
    else:
        # Sleek Search Filter at the top of the Vault Grid with dynamic Clear button
        if "search_key_index" not in st.session_state:
            st.session_state.search_key_index = 0

        search_key = f"vault_search_input_{st.session_state.search_key_index}"

        search_col1, search_col2 = st.columns([5, 1])
        with search_col1:
            search_query = st.text_input(
                "🔍 Search Audited Contracts Vault",
                placeholder="Search by contract name...",
                label_visibility="collapsed",
                key=search_key,
            )

        with search_col2:
            if search_query:
                if st.button("❌ Clear Search", use_container_width=True):
                    st.session_state.search_key_index += 1
                    st.rerun()

        if search_query:
            filtered_completed = [
                item for item in completed if search_query.lower() in item["id"].lower()
            ]
        else:
            filtered_completed = completed

        if not filtered_completed:
            st.info(
                f"🔍 No audited contracts matched your search query: '{search_query}'"
            )
        else:
            # Render responsive 2-column grid
            cols = st.columns(2)
            for idx, item in enumerate(filtered_completed):
                col_target = cols[idx % 2]

                with col_target:
                    score = item["score"]
                    score_percentage = f"{int(score * 100)}%"

                    if score >= 0.5:
                        badge_color = "badge-high"
                        badge_label = "HIGH RISK"
                    elif score >= 0.25:
                        badge_color = "badge-medium"
                        badge_label = "MODERATE"
                    else:
                        badge_color = "badge-low"
                        badge_label = "SECURE"

                    date_str = datetime.fromtimestamp(item["mtime"]).strftime(
                        "%B %d, %Y at %I:%M %p"
                    )

                    # Render the entire vault card wrapped in a beautiful web-native anchor tag!
                    st.markdown(
                        f"""
                    <a href="?doc_id={item['id']}" target="_self" style="text-decoration: none; color: inherit; display: block;">
                        <div class="vault-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                                <span class="vault-card-title">{item['id']}</span>
                                <span class="badge {badge_color}">{badge_label}</span>
                            </div>
                            <div class="vault-card-date">🗓️ Audited on {date_str}</div>
                            <div class="vault-stats-row">
                                <span>📊 <strong>Overall Risk:</strong> <span style="font-weight: 700;">{score_percentage}</span></span>
                                <span>📄 <strong>Clauses:</strong> <span style="font-weight: 700;">{item['total_clauses']}</span></span>
                                <span>⚠️ <strong>Missing:</strong> <span style="font-weight: 700; color:{'#DC2626' if item['missing'] > 0 else '#059669'};">{item['missing']}</span></span>
                            </div>
                        </div>
                    </a>
                    """,
                        unsafe_allow_html=True,
                    )

else:
    # INDIVIDUAL REPORT AUDITED VIEW
    doc_dir = PROJECT_ROOT / "data" / "output" / "final" / selected_doc
    report_json_path = doc_dir / "final_report.json"

    if not report_json_path.exists():
        st.markdown(
            f"<div class='banner-title'>⚠️ Suspended Audit Session</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='banner-subtitle'>Contract: {selected_doc}</div>",
            unsafe_allow_html=True,
        )

        st.write("")
        st.warning(
            "**Audit Session Suspended**: This contract compliance audit was started but was suspended or interrupted before completion. The checkpointer has safely cached all intermediate results."
        )

        res_col1, res_col2 = st.columns([1, 3])
        with res_col1:
            if st.button("👈 Back to Vault Dashboard", use_container_width=True):
                st.query_params.clear()
                st.session_state.selected_doc = None
                st.rerun()
        with res_col2:
            if st.button("🔄 Resume and Complete Audit Now", use_container_width=True):
                st.session_state["_do_resume"] = selected_doc

        # Execute pipeline OUTSIDE the column context so the timeline renders full-width
        if st.session_state.get("_do_resume") == selected_doc:
            st.session_state.pop("_do_resume")
            raw_path = PROJECT_ROOT / "data" / "raw" / f"{selected_doc}.pdf"
            stage1_cache = PROJECT_ROOT / "data" / "output" / "final" / selected_doc / "stage1_output.json"
            
            if not raw_path.exists() and not stage1_cache.exists():
                st.error(
                    f"Raw source PDF not found at {raw_path}. Please go back and re-upload the contract."
                )
            else:
                execute_stepped_pipeline(raw_path, selected_doc)
                st.rerun()
    else:
        with open(report_json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Show background audit alert if active
        bg_audit = st.session_state.get("active_audit")
        if bg_audit and bg_audit != selected_doc:
            st.info(
                f"🧠 **Active Background Audit**: An audit is currently running in the background for **{bg_audit}**. You can safely navigate the system; click **Back to Vault Dashboard** below anytime to return to its live progress timeline!"
            )

        # Top Navigation Bar (100% Horizontal Back Button!)
        nav_col1, nav_col2 = st.columns([1, 4])
        with nav_col1:
            if st.button("👈 Back to Vault Dashboard", use_container_width=True):
                st.query_params.clear()
                st.session_state.selected_doc = None
                st.rerun()
        st.write("")

        # Top-level Document Banner
        st.markdown(
            f"<div class='banner-title'>⚖️ {selected_doc}</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='banner-subtitle'>Interactive Compliance Audit Report</div>",
            unsafe_allow_html=True,
        )

        # Top Metric Cards
        col1, col2, col3 = st.columns(3)
        score = report.get("overall_risk_score", 0.0)
        score_percentage = f"{int(score * 100)}%"

        if score >= 0.5:
            score_color = "#DC2626"  # Crimson
            score_bg = "#FEF2F2"
        elif score >= 0.25:
            score_color = "#D97706"  # Amber
            score_bg = "#FFFBEB"
        else:
            score_color = "#059669"  # Emerald Green
            score_bg = "#ECFDF5"

        with col1:
            st.markdown(
                f"""
            <div class="metric-card" style="border-color: {score_color}60; background-color: {score_bg};">
                <div class="metric-label" style="color: {score_color}">Overall Risk Index</div>
                <div class="metric-value" style="color: {score_color}">{score_percentage}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Audited Clauses</div>
                <div class="metric-value">{report.get("total_clauses", 0)}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            missing_count = len(report.get("missing_protections", []))
            missing_color = "#DC2626" if missing_count > 0 else "#0F172A"
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Missing Standard Protections</div>
                <div class="metric-value" style="color: {missing_color}">{missing_count}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.write("")

        # Missing Protections alert banner (CUAD wording fully cleaned out)
        missing_protections = report.get("missing_protections", [])
        if missing_protections:
            st.warning(
                f"⚠️ **Essential Legal Protections Absent:** "
                f"This contract completely lacks standard safeguarding provisions for: `{', '.join(missing_protections)}`"
            )
        else:
            st.success(
                "🎉 **Pristine Contract Compliance:** All standard protective clauses are fully present in this contract!"
            )

        st.write("")

        # Highly Intuitive Main Page Horizontal Tabs (Enlarged & Presentable via CSS!)
        tab_summary, tab_high, tab_med, tab_low, tab_all = st.tabs(
            [
                "📝 Executive Synthesis",
                "🔴 High Risk Issues",
                "🟡 Moderate Risk Issues",
                "🟢 Safe / Low Risk Issues",
                "📄 Compile Markdown Report",
            ]
        )

        def render_clause_cards(clause_list, risk_type):
            if not clause_list:
                st.info(
                    f"No {risk_type.lower()} compliance concerns identified in this contract."
                )
                return

            for idx, clause in enumerate(clause_list, 1):
                badge_class = f"badge-{risk_type.lower()}"
                confidence = clause.get("risk_confidence", 0.0)
                conf_badge = (
                    f"<span class='badge' style='background-color: #F1F5F9; color: #475569; margin-right: 8px;'>Confidence: {int(confidence*100)}%</span>"
                    if confidence > 0
                    else ""
                )

                st.markdown(
                    f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);">
                    <div class="clause-card-header">
                        <span class="clause-card-title">{idx}. {clause['clause_type']}</span>
                        <div>
                            {conf_badge}
                            <span class="badge {badge_class}">{risk_type} Risk</span>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                clause_text = clause.get("clause_text")
                page_no = clause.get("page_no")
                ref_page_str = f" (Ref Page {page_no})" if page_no else ""

                if clause_text:
                    st.markdown(
                        f"""
                    <div style="background-color: #F8FAFC; border-left: 4px solid #94A3B8; padding: 14px 18px; border-radius: 6px; font-style: italic; color: #334155; margin-bottom: 16px; font-size: 1rem; line-height: 1.6;">
                        <strong style="color: #475569;">Original Clause Text{ref_page_str}:</strong><br>
                        "{clause_text}"
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Compliance Assessment Brief:** {clause['explanation']}")

                rec = clause.get("recommendation")
                if rec:
                    if risk_type == "HIGH":
                        rec_border = "#FCA5A5"
                        rec_bg = "#FEF2F2"
                        rec_color = "#991B1B"
                    elif risk_type == "MEDIUM":
                        rec_border = "#FCD34D"
                        rec_bg = "#FFFBEB"
                        rec_color = "#92400E"
                    else:
                        rec_border = "#6EE7B7"
                        rec_bg = "#ECFDF5"
                        rec_color = "#065F46"

                    st.markdown(
                        f"""
                    <div style="background-color: {rec_bg}; border: 1px solid {rec_border}; border-radius: 8px; padding: 14px 18px; color: {rec_color}; font-size: 0.98rem; margin-top: 12px; margin-bottom: 16px; line-height: 1.5;">
                        <strong>💡 Actionable Recommendation:</strong> {rec}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                similar = clause.get("similar_clauses", [])
                cross_refs = clause.get("cross_references", [])
                if similar or cross_refs:
                    with st.expander(
                        "📚 View Gold-Standard Precedents & Internal References"
                    ):
                        exp_col1, exp_col2 = st.columns(2)

                        with exp_col1:
                            st.markdown(
                                "<p style='font-size: 0.95rem; font-weight: 700; color: #1F2937; margin-bottom: 12px;'>🌟 Market Precedents (Comparative Ground Truth)</p>",
                                unsafe_allow_html=True,
                            )
                            if similar:
                                for s in similar[:3]:  # Show top 3 for optimal space
                                    prec_risk = s.get(
                                        "risk_level", s.get("risk", "LOW")
                                    ).upper()
                                    prec_similarity = int(
                                        s.get("similarity", 0.0) * 100
                                    )
                                    prec_text = s.get("text", "")

                                    if prec_risk == "HIGH":
                                        prec_badge_class = "badge-high"
                                    elif (
                                        prec_risk == "MEDIUM" or prec_risk == "MODERATE"
                                    ):
                                        prec_badge_class = "badge-medium"
                                    else:
                                        prec_badge_class = "badge-low"

                                    st.markdown(
                                        f"""
                                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 8px; font-size: 0.88rem;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                            <strong style="color: #1E3C72; font-size: 0.85rem;">📋 {s.get('clause_type', 'Precedent')}</strong>
                                            <div>
                                                <span class="badge {prec_badge_class}" style="font-size: 0.7rem; padding: 2px 6px;">{prec_risk}</span>
                                                <span style="font-size: 0.7rem; font-weight: 700; color: #475569; background-color: #E2E8F0; padding: 2px 6px; border-radius: 9999px; margin-left: 4px;">{prec_similarity}% Sim</span>
                                            </div>
                                        </div>
                                        <div style="color: #475569; font-style: italic; line-height: 1.45;">
                                            "{prec_text}"
                                        </div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info("No comparative market precedents identified.")

                        with exp_col2:
                            st.markdown(
                                "<p style='font-size: 0.95rem; font-weight: 700; color: #1F2937; margin-bottom: 12px;'>🔗 Internal Cross-References & Context</p>",
                                unsafe_allow_html=True,
                            )
                            if cross_refs:
                                for ref in cross_refs[
                                    :3
                                ]:  # Show top 3 for optimal space
                                    if isinstance(ref, dict):
                                        ref_type = ref.get("clause_type", "Reference")
                                        ref_text = ref.get("clause_text", "")
                                    else:
                                        ref_type = "Related Section"
                                        ref_text = str(ref)

                                    st.markdown(
                                        f"""
                                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; margin-bottom: 8px; font-size: 0.88rem;">
                                        <div style="margin-bottom: 6px;">
                                            <strong style="color: #0369A1; font-size: 0.85rem;">🔗 {ref_type}</strong>
                                        </div>
                                        <div style="color: #475569; font-style: italic; line-height: 1.45;">
                                            "{ref_text[:300]}..."
                                        </div>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info(
                                    "No internal cross-references parsed in surrounding context."
                                )

                trace = clause.get("agent_trace", [])
                if trace:
                    with st.expander("🔍 View AI Reasoning Path"):
                        steps = []
                        for t in trace:
                            t_tool = t.get("tool", "Unknown")
                            t_hits = t.get("result_count", 0)

                            # Custom professional mappings
                            name_lower = t_tool.lower()
                            if "contract" in name_lower or "extract" in name_lower:
                                mapped_name = "Extracting AI"
                            elif "precedent" in name_lower or "faiss" in name_lower:
                                mapped_name = "Analyzing AI"
                            elif "ambiguity" in name_lower or "resolve" in name_lower:
                                mapped_name = "Classifying AI"
                            else:
                                mapped_name = "Analyzing AI"

                            steps.append(f"**{mapped_name}** ({t_hits} hits)")
                        st.markdown(
                            " &nbsp;➔&nbsp; ".join(steps), unsafe_allow_html=True
                        )

                st.write("")
                st.write("")

        with tab_summary:
            st.markdown("### 📝 Executive Synthesis")
            st.markdown(
                f"<div style='font-size: 1.12rem; line-height: 1.75; color: #1E293B; padding: 24px; background-color: #F8FAFC; border-radius: 12px; border-left: 5px solid #2563EB;'>{report.get('summary', 'No synthesis briefing generated.')}</div>",
                unsafe_allow_html=True,
            )

            # Timeline & Telemetry Panel (HTML Badges - Highly Legible and Presentable!)
            st.markdown("---")
            st.markdown("#### ⚙️ Compliance Telemetry & Audit Footprint")
            meta = report.get("metadata", {})
            models = meta.get("models_used", {})

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                st.markdown(
                    f"<div style='font-size: 1.05rem; margin-bottom: 12px;'><strong>Audit Timestamp:</strong> &nbsp;&nbsp;{format_datetime_beautiful(meta.get('generated_at', ''))}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size: 1.05rem; margin-bottom: 12px;'><strong>Internal Document ID:</strong> &nbsp;&nbsp;<span class='telemetry-badge-blue'>{report.get('document_id')}</span></div>",
                    unsafe_allow_html=True,
                )
            with tcol2:
                st.markdown(
                    "<div style='font-size: 1.05rem; margin-bottom: 12px;'><strong>Document Structural Parser:</strong> &nbsp;&nbsp;<span class='telemetry-badge-blue'>Extracting AI</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size: 1.05rem; margin-bottom: 12px;'><strong>Risk Inference Classifier:</strong> &nbsp;&nbsp;<span class='telemetry-badge'>Classifying AI</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div style='font-size: 1.05rem; margin-bottom: 12px;'><strong>Executive Synthesis Reasoner:</strong> &nbsp;&nbsp;<span class='telemetry-badge-purple'>Analyzing AI</span></div>",
                    unsafe_allow_html=True,
                )

        with tab_high:
            st.markdown("### 🔴 High Risk Compliance Concerns")
            render_clause_cards(report.get("high_risk", []), "HIGH")

        with tab_med:
            st.markdown("### 🟡 Moderate Risk Compliance Concerns")
            render_clause_cards(report.get("medium_risk", []), "MEDIUM")

        with tab_low:
            st.markdown("### 🟢 Safe / Low Risk Protections")
            render_clause_cards(report.get("low_risk", []), "LOW")

        with tab_all:
            st.markdown("### 📄 Full Contract Analysis Report (Markdown Format)")

            from src.common.schema import (
                RiskReport,
                ReportClause,
                ReportMetadata,
                SimilarClause,
            )

            def to_report_clause_obj(c):
                sims = [
                    SimilarClause(
                        text=s.get("text", ""),
                        clause_type=s.get("clause_type", "Precedent"),
                        risk_level=s.get("risk_level", s.get("risk", "LOW")),
                        similarity=s.get("similarity", 0.0),
                    )
                    for s in c.get("similar_clauses", [])
                ]
                return ReportClause(
                    clause_id=c.get("clause_id", ""),
                    clause_type=c.get("clause_type", ""),
                    risk_level=c.get("risk_level", "LOW"),
                    explanation=c.get("explanation", ""),
                    recommendation=c.get("recommendation", ""),
                    similar_clauses=sims,
                    cross_references=c.get("cross_references", []),
                    page_no=c.get("page_no"),
                    risk_confidence=c.get("risk_confidence", 0.0),
                    clause_text=c.get("clause_text", ""),
                )

            high_objs = [to_report_clause_obj(c) for c in report.get("high_risk", [])]
            med_objs = [to_report_clause_obj(c) for c in report.get("medium_risk", [])]
            low_objs = [to_report_clause_obj(c) for c in report.get("low_risk", [])]

            report_obj = RiskReport(
                document_id=report.get("document_id", ""),
                summary=report.get("summary", ""),
                high_risk=high_objs,
                medium_risk=med_objs,
                low_risk=low_objs,
                low_risk_summary=report.get("low_risk_summary", ""),
                missing_protections=report.get("missing_protections", []),
                overall_risk_score=report.get("overall_risk_score", 0.0),
                total_clauses=report.get("total_clauses", 0),
                metadata=ReportMetadata(
                    generated_at=report.get("metadata", {}).get("generated_at", ""),
                    models_used=report.get("metadata", {}).get("models_used", {}),
                ),
            )

            md_content = export_markdown_report(report_obj)
            st.info(
                "💡 **Single-Click Copy**: Click the copy icon in the top-right of the box below to instantly copy the raw markdown to your clipboard."
            )
            st.code(md_content, language="markdown")

            st.markdown("#### Live Rendered Preview")
            st.markdown(md_content)
