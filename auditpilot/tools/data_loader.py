"""
auditpilot/tools/data_loader.py - Secure data loader for AuditPilot agents.
Strictly blocks access to ground_truth.csv and environment secrets.
Implements load_ledger with comprehensive schema and integrity validations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

BLOCKED_FILES = {"ground_truth.csv", ".env"}

DEFAULT_LEDGER_COLUMNS = [
    "entry_id",
    "date",
    "posted_by",
    "approved_by",
    "account_code",
    "account_name",
    "vendor_id",
    "amount_inr",
    "description",
]


def load_table(table_name: str | Path, data_dir: str = "data") -> pd.DataFrame:
    """
    Safely load a dataset table for audit analysis.
    Raises PermissionError if an agent or tool attempts to access restricted files.
    """
    path = Path(table_name)
    if not path.is_absolute() and len(path.parts) == 1:
        path = Path(data_dir) / table_name

    normalized_name = path.name.lower()
    full_str = str(path).lower()

    if normalized_name in BLOCKED_FILES or "ground_truth" in normalized_name or "ground_truth" in full_str:
        raise PermissionError(
            f"Access to '{path.name}' is strictly forbidden for audit agents. "
            "Audit tools and agents cannot access ground truth benchmarks."
        )

    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{path}' does not exist.")

    return pd.read_csv(path)


def list_accessible_tables(data_dir: str = "data") -> list[str]:
    """
    List all tables available to audit agents, omitting restricted files.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        return []

    accessible = []
    for f in sorted(dir_path.iterdir()):
        normalized_name = f.name.lower()
        if f.is_file() and normalized_name not in BLOCKED_FILES and "ground_truth" not in normalized_name:
            accessible.append(f.name)
    return accessible


def load_ledger(
    file_path: str | Path = "data/ledger.csv",
    required_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load and validate financial ledger records.
    
    Parameters:
        file_path: Path to the ledger CSV file.
        required_columns: Optional list of required column names.
        
    Returns:
        Standard audit tool dictionary containing:
            - test: "load_ledger"
            - flagged_ids: List of entry_ids with validation anomalies or missing fields
            - stats: Summary statistics of the loaded ledger
            - notes: Explanatory audit observations
            - data: The validated pandas DataFrame
    """
    if required_columns is None:
        required_columns = DEFAULT_LEDGER_COLUMNS

    df = load_table(file_path)

    # 1. Column existence verification
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Ledger file '{file_path}' is missing required columns: {missing_cols}")

    flagged_ids = []
    notes = []

    # 2. Missing value checks
    critical_cols = ["entry_id", "date", "posted_by", "approved_by", "amount_inr"]
    null_mask = df[critical_cols].isna().any(axis=1)
    if null_mask.any():
        null_ids = df.loc[null_mask, "entry_id"].dropna().tolist()
        flagged_ids.extend(null_ids)
        notes.append(f"Found {len(null_ids)} records with missing critical fields: {null_ids[:5]}.")

    # 3. Data type and range validations
    # Validate amounts are numeric and positive
    invalid_amounts = df[~pd.to_numeric(df["amount_inr"], errors="coerce").notna() | (df["amount_inr"] <= 0)]
    if not invalid_amounts.empty:
        amt_ids = invalid_amounts["entry_id"].tolist()
        flagged_ids.extend(amt_ids)
        notes.append(f"Found {len(amt_ids)} records with non-positive or invalid amounts.")

    # Validate date parsing
    invalid_dates = df[~pd.to_datetime(df["date"], errors="coerce").notna()]
    if not invalid_dates.empty:
        date_ids = invalid_dates["entry_id"].tolist()
        flagged_ids.extend(date_ids)
        notes.append(f"Found {len(date_ids)} records with malformed date strings.")

    # Deduplicate flagged IDs while preserving order
    unique_flagged_ids = list(dict.fromkeys(flagged_ids))

    total_records = len(df)
    valid_records = total_records - len(unique_flagged_ids)
    date_series = pd.to_datetime(df["date"], errors="coerce").dropna()
    min_date = date_series.min().strftime("%Y-%m-%d") if not date_series.empty else "N/A"
    max_date = date_series.max().strftime("%Y-%m-%d") if not date_series.empty else "N/A"
    total_amount = round(float(pd.to_numeric(df["amount_inr"], errors="coerce").sum()), 2)

    if not notes:
        notes.append(f"Successfully loaded and validated {total_records:,} ledger records with complete schema conformity.")

    return {
        "test": "load_ledger",
        "flagged_ids": unique_flagged_ids,
        "stats": {
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": len(unique_flagged_ids),
            "date_range": [min_date, max_date],
            "total_amount_inr": total_amount,
        },
        "notes": notes,
        "data": df,
    }
