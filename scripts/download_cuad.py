"""Download and cache the CUAD dataset from Zenodo and extract test-split contracts.

Usage:
    uv run python scripts/download_cuad.py [--output data/raw/cuad_test_contract] [--count 5]
"""

import argparse
import logging
import os
import json
import zipfile
import subprocess
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_cuad")

ZENODO_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip"
CACHE_ZIP_PATH = Path("data/raw/CUAD_v1.zip")
SPLITS_PATH = Path("data/processed/splits.json")
DATASET_PATH = Path("data/processed/training_dataset.json")


def download_file_with_progress(url: str, dest_path: Path):
    """Download a file using system curl to bypass Cloudflare/Zenodo bot blocks."""
    logger.info("Initializing download from %s using system curl...", url)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run native system curl command which is pre-authorized by Zenodo
    # -L: Follow redirects
    # --progress-bar: Show clear percentage bar
    # -o: Save to destination path
    cmd = ["curl", "-L", "--progress-bar", "-o", str(dest_path), url]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Successfully downloaded CUAD dataset archive to %s", dest_path)
    except subprocess.CalledProcessError as e:
        logger.error("Download failed via curl: %s", e)
        raise


def download_cuad(output_dir: str = "data/raw/cuad_test_contract", count: int = 5) -> None:
    """Download the CUAD dataset and extract a specific number of PDFs from the test split."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the test split contract names
    if not SPLITS_PATH.exists() or not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Required split files ({SPLITS_PATH} or {DATASET_PATH}) do not exist. "
            "Please ensure you are running from the workspace root."
        )
        
    logger.info("Loading test split contract lists...")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
        
    dataset_by_row = {r["row_num"]: r for r in dataset}
    test_rows = splits.get("test", [])
    
    # Get unique contract names from the test split
    test_contract_names = sorted(list(set(
        dataset_by_row[row]["contract"] 
        for row in test_rows 
        if row in dataset_by_row
    )))
    
    logger.info("Identified %d unique contract names in the test split.", len(test_contract_names))
    
    # We will pick the top 'count' unique contracts to extract
    selected_contracts = test_contract_names[:count]
    logger.info("Target contracts selected for extraction (%d):", len(selected_contracts))
    for idx, name in enumerate(selected_contracts, 1):
        logger.info("  %d. %s", idx, name)

    # 2. Check/Download the main zip file
    if not CACHE_ZIP_PATH.exists():
        logger.info("Local CUAD Zip archive not found. Fetching from Zenodo...")
        download_file_with_progress(ZENODO_URL, CACHE_ZIP_PATH)
    else:
        logger.info("Found existing local CUAD Zip archive at %s. Skipping download.", CACHE_ZIP_PATH)

    # 3. Open the Zip and extract matching PDFs
    logger.info("Opening CUAD Zip archive to extract selected contracts...")
    extracted_files = []
    
    # Helper to robustly normalize filenames (handling URL encoding, spacing, underscores, casing, etc.)
    import urllib.parse
    def normalize_name(s: str) -> str:
        decoded = urllib.parse.unquote(s)
        return "".join(c for c in decoded.lower() if c.isalnum())

    with zipfile.ZipFile(CACHE_ZIP_PATH, "r") as z:
        # List all files in the zip to find the PDFs
        all_files = z.namelist()
        # Scan for full_contract_pdf (singular contract, nested directories)
        pdf_files = [f for f in all_files if "full_contract_pdf" in f and f.lower().endswith(".pdf")]
        
        logger.info("Scanned zip file. Found %d total PDF contracts in archive.", len(pdf_files))
        
        # Build a mapping from normalized base contract name -> full zip path
        pdf_map = {}
        for zip_path in pdf_files:
            base_name = Path(zip_path).stem
            norm_base = normalize_name(base_name)
            pdf_map[norm_base] = zip_path
            
        # Match our selected test split names with the zip paths
        for idx, contract_name in enumerate(selected_contracts, 1):
            norm_contract = normalize_name(contract_name)
            matched_zip_path = None
            
            if norm_contract in pdf_map:
                matched_zip_path = pdf_map[norm_contract]
            else:
                # Fallback: substring matching in case of slight suffix or prefix mismatches
                for k, v in pdf_map.items():
                    if norm_contract in k or k in norm_contract:
                        matched_zip_path = v
                        logger.info("Soft-matched contract: '%s' with archive entry: '%s'", contract_name, Path(v).name)
                        break
            
            if matched_zip_path:
                # Clean up filename for saving
                clean_filename = urllib.parse.unquote(contract_name).replace("/", "_").replace("%20", " ")
                if not clean_filename.endswith(".pdf"):
                    clean_filename += ".pdf"
                dest_file_path = output_path / clean_filename
                
                logger.info("[%d/%d] Extracting: %s -> %s", 
                            idx, len(selected_contracts), contract_name, dest_file_path)
                
                # Extract the file
                with z.open(matched_zip_path) as source, open(dest_file_path, "wb") as target:
                    target.write(source.read())
                    
                extracted_files.append(dest_file_path)
            else:
                logger.warning("Could not find matching PDF in archive for contract: %s", contract_name)

    # 4. Success summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD & EXTRACTION COMPLETED SUCCESSFULLY!")
    logger.info("Extracted %d test split contract PDFs into: %s", len(extracted_files), output_path)
    logger.info("=" * 60)
    logger.info("To run the end-to-end risk pipeline on one of your new contracts, execute:")
    if extracted_files:
        sample_path = extracted_files[0]
        logger.info("  rm data/checkpoints/agent_state.db")
        logger.info("  uv run python scripts/run_pipeline.py --contract \"%s\"", sample_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract test split contracts from CUAD.")
    parser.add_argument("--output", default="data/raw/cuad_test_contract", 
                        help="Target output directory for the extracted PDFs.")
    parser.add_argument("--count", type=int, default=5, 
                        help="Number of unique test-split contracts to extract (default: 5).")
    args = parser.parse_args()
    
    download_cuad(output_dir=args.output, count=args.count)
