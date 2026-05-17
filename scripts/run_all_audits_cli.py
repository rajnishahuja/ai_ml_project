import os
import sys
import time
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_auditor")

from src.common.pipeline_service import run_end_to_end_pipeline

def main():
    final_dir = PROJECT_ROOT / "data" / "output" / "final"
    
    # Get all active contract subdirectories in final output
    contract_dirs = [
        d for d in final_dir.iterdir() 
        if d.is_dir() and not d.name.endswith("_backup")
    ]
    
    print("\n" + "="*70)
    print("      🚀 LEGAL CONTRACT RISK ANALYZER - BATCH AUDIT CLI")
    print("="*70)
    print(f"Found {len(contract_dirs)} contracts in the output directory.")
    for d in contract_dirs:
        print(f"  • {d.name}")
    print("="*70)

    results = []
    
    for d in contract_dirs:
        doc_id = d.name
        print(f"\n⚡ Starting audit for: {doc_id}")
        
        # We pass a dummy path because the Stage 1 & 2 caches exist,
        # so pdfplumber/docling conversion is completely bypassed!
        dummy_path = str(PROJECT_ROOT / "data" / "raw" / f"{doc_id}.pdf")
        
        # Ensure a dummy raw PDF directory exists just in case
        os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
        
        start_time = time.perf_counter()
        try:
            report = run_end_to_end_pipeline(
                contract_path=dummy_path,
                doc_id=doc_id,
                stage1_model="rajnishahuja/cuad-stage1-deberta",
            )
            duration = time.perf_counter() - start_time
            
            high_count = len(report.high_risk)
            med_count = len(report.medium_risk)
            low_count = report.total_clauses - high_count - med_count
            
            results.append({
                "doc_id": doc_id,
                "status": "SUCCESS",
                "score": f"{report.overall_risk_score:.2f}",
                "high": high_count,
                "med": med_count,
                "low": low_count,
                "time": f"{duration:.2f}s"
            })
            print(f"✅ SUCCESS: {doc_id} (Score: {report.overall_risk_score:.2f}, High: {high_count}, Med: {med_count}) in {duration:.2f}s")
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            print(f"❌ FAILED: {doc_id} in {duration:.2f}s")
            print(f"   Error: {e}")
            results.append({
                "doc_id": doc_id,
                "status": f"FAILED: {str(e)[:50]}",
                "score": "N/A",
                "high": 0,
                "med": 0,
                "low": 0,
                "time": f"{duration:.2f}s"
            })

    # Print gorgeous batch summary report
    print("\n" + "="*95)
    print("                                BATCH AUDIT EXECUTION SUMMARY")
    print("="*95)
    print(f"{'Contract/Document ID':<45} | {'Status':<10} | {'Score':<6} | {'High':<5} | {'Med':<5} | {'Time':<7}")
    print("-"*95)
    for r in results:
        print(f"{r['doc_id'][:45]:<45} | {r['status']:<10} | {r['score']:<6} | {r['high']:<5} | {r['med']:<5} | {r['time']:<7}")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
