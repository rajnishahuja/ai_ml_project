import os
import shutil
import json
import time
from pathlib import Path
from src.common.pipeline_service import run_end_to_end_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_ID = "CORIOINC_07_20_2000-EX-10.5-LICENSE AND HOSTING AGREEMENT"
RAW_PDF = str(PROJECT_ROOT / "data" / "raw" / f"{DOC_ID}.pdf")
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "final" / DOC_ID
BACKUP_DIR = PROJECT_ROOT / "data" / "output" / "final" / f"{DOC_ID}_backup"

def backup_original_outputs():
    """Backup all original complete stage outputs to preserve them."""
    if OUTPUT_DIR.exists():
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(OUTPUT_DIR, BACKUP_DIR)
        print("Backup created successfully.")

def restore_from_backup(keep_stages: list[int]):
    """
    Restore only specific stage files from backup to test recovery at that point.
    keep_stages is a list of stages to preserve (e.g. [1] means preserve only Stage 1 files).
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Restore specific files based on keep_stages
    if 1 in keep_stages:
        shutil.copy(BACKUP_DIR / "stage1_output.json", OUTPUT_DIR / "stage1_output.json")
        shutil.copy(BACKUP_DIR / "stage1_layout.txt", OUTPUT_DIR / "stage1_layout.txt")
        print("  -> Kept Stage 1 files")
        
    if 2 in keep_stages:
        shutil.copy(BACKUP_DIR / "stage2_output.json", OUTPUT_DIR / "stage2_output.json")
        print("  -> Kept Stage 2 files")
        
    if 3 in keep_stages:
        shutil.copy(BACKUP_DIR / "stage3_output.json", OUTPUT_DIR / "stage3_output.json")
        print("  -> Kept Stage 3 files")

def restore_full_backup():
    """Restore the complete original folder after all tests finish."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if BACKUP_DIR.exists():
        shutil.copytree(BACKUP_DIR, OUTPUT_DIR)
        shutil.rmtree(BACKUP_DIR)
        print("Full backup restored.")

def run_test_case(name: str, keep_stages: list[int], expected_bypassed_stages: list[int]):
    print(f"\n==================================================")
    print(f"RUNNING: {name}")
    print(f"==================================================")
    
    # 1. Restore the directory with only the selected stage caches
    restore_from_backup(keep_stages)
    
    # 2. Run the pipeline
    start_time = time.perf_counter()
    run_end_to_end_pipeline(RAW_PDF, DOC_ID, stage1_model='rajnishahuja/cuad-stage1-deberta')
    duration = time.perf_counter() - start_time
    
    # 3. Read generated latency metrics
    metrics_path = OUTPUT_DIR / "latency_metrics.json"
    assert metrics_path.exists(), "Error: latency_metrics.json was not created!"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    # 4. Check bypass assertions
    print(f"Latency Results (Total: {duration:.2f}s):")
    
    # Assert Stage 1
    s1_time = metrics["stage1_layout_parsing_seconds"]
    if 1 in expected_bypassed_stages:
        print(f"  Stage 1: Bypassed successfully (cached duration: {s1_time:.4f}s)")
        assert s1_time < 0.5, f"Stage 1 should have been bypassed, but took {s1_time}s"
    else:
        print(f"  Stage 1: Executed (duration: {s1_time:.4f}s)")
        
    # Assert Stage 2
    s2_time = metrics["stage2_span_extraction_seconds"]
    if 2 in expected_bypassed_stages:
        print(f"  Stage 2: Bypassed successfully (cached duration: {s2_time:.4f}s)")
        assert s2_time < 0.5, f"Stage 2 should have been bypassed, but took {s2_time}s"
    else:
        print(f"  Stage 2: Executed (duration: {s2_time:.4f}s)")
        
    # Assert Stage 3
    s3_time = metrics["stage3_risk_assessment_seconds"]
    if 3 in expected_bypassed_stages:
        print(f"  Stage 3: Bypassed successfully (cached duration: {s3_time:.4f}s)")
        assert s3_time < 0.5, f"Stage 3 should have been bypassed, but took {s3_time}s"
    else:
        print(f"  Stage 3: Executed (duration: {s3_time:.4f}s)")
        
    # Assert Stage 4 (build report is extremely fast but always runs)
    s4_time = metrics["stage4_report_generation_seconds"]
    print(f"  Stage 4: Executed (duration: {s4_time:.4f}s)")
    
    # Verify all final report files exist
    assert (OUTPUT_DIR / "stage1_output.json").exists()
    assert (OUTPUT_DIR / "stage2_output.json").exists()
    assert (OUTPUT_DIR / "stage3_output.json").exists()
    assert (OUTPUT_DIR / "final_report.json").exists()
    assert (OUTPUT_DIR / "report.html").exists()
    print(f"SUCCESS: {name} completed flawlessly!")

if __name__ == "__main__":
    backup_original_outputs()
    try:
        # Test Case A: Keep only Stage 1 (resumes from Stage 2)
        run_test_case(
            name="Test A: Resume from Stage 1 Cache (Stage 1 bypassed, Stages 2-4 run)",
            keep_stages=[1],
            expected_bypassed_stages=[1]
        )
        
        # Test Case B: Keep Stage 1 & Stage 2 (resumes from Stage 3)
        run_test_case(
            name="Test B: Resume from Stage 2 Cache (Stages 1 & 2 bypassed, Stages 3-4 run)",
            keep_stages=[1, 2],
            expected_bypassed_stages=[1, 2]
        )
        
        # Test Case C: Keep Stage 1, Stage 2, & Stage 3 (resumes from Stage 4)
        run_test_case(
            name="Test C: Resume from Stage 3 Cache (Stages 1, 2, & 3 bypassed, only Stage 4 runs)",
            keep_stages=[1, 2, 3],
            expected_bypassed_stages=[1, 2, 3]
        )
        
    finally:
        restore_full_backup()
