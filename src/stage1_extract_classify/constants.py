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