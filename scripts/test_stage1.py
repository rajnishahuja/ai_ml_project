import json
import argparse
import logging
from pathlib import Path
from src.stage1_extract_classify.preprocessing import preprocess_contract
from src.stage1_extract_classify.model import ClauseExtractorClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, help="Path to PDF")
    parser.add_argument("--output", default="data/output/stage1/stage1_output.json", help="Path to save JSON")
    parser.add_argument("--model", default="rajnishahuja/cuad-stage1-deberta")
    args = parser.parse_args()

    doc_id = Path(args.contract).stem
    
    logger.info("Stage 1: Preprocessing PDF...")
    text = preprocess_contract(args.contract, doc_id)
    
    logger.info("Stage 1: Running DeBERTa Extraction...")
    extractor = ClauseExtractorClassifier(args.model)
    clauses = extractor.extract(text, doc_id)
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save as JSON list
    output_data = [c.to_dict() for c in clauses]
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Done! Extracted {len(clauses)} clauses. saved to {args.output}")

if __name__ == "__main__":
    main()
