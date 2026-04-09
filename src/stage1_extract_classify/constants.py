# ---------------------------------------------------------------------------
# CUAD clause types (41 categories from the dataset)
# ---------------------------------------------------------------------------
CUAD_CLAUSE_TYPES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "ROFR/ROFO/ROFN",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
    "Indemnification",
]

# CUAD question templates (one per clause type, matching the dataset format)
CUAD_QUESTION_TEMPLATES = {
    clause: f"Highlight the parts (if any) of this contract related to \"{clause}\" that should be reviewed by a lawyer. Details: {clause}"
    for clause in CUAD_CLAUSE_TYPES
}

# add a reverse lookup dict
QUESTION_TO_CLAUSE_TYPE = {
    v: k for k, v in CUAD_QUESTION_TEMPLATES.items()
}

BASELINE_CONF_THRESHOLD = 0.6

import os
from datasets import load_from_disk
import logging
logger = logging.getLogger(__name__)


def get_data_path(override: str = None) -> str:
    """
    Resolve dataset path based on environment.
    Priority: explicit override > env var > auto-detected platform default.
    """
    if override:
        return override

    if env_path := os.environ.get("CUAD_DATA_PATH"):
        return env_path

    # Kaggle — datasets are mounted under /kaggle/
    if os.path.exists("/kaggle/working"):
        return "/kaggle/input/tokenized-cuad/tokenized_cuad"

    # Colab — typically mounted from Drive
    if os.path.exists("/content/drive"):
        return "/content/drive/MyDrive/aiml_project/tokenized_cuad"

    # Local fallback
    return "./data/tokenized_cuad"


def load_cuad_dataset(data_path: str = None):
    path = get_data_path(data_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Set CUAD_DATA_PATH env var or pass data_path explicitly.\n"
            f"Example: load_cuad_dataset('/your/actual/path')"
        )
    logger.info(f"Loading dataset from: {path}")
    return load_from_disk(path)