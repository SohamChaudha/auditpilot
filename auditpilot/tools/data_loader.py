"""
auditpilot/tools/data_loader.py - Secure data loader for AuditPilot agents.
Strictly blocks access to ground_truth.csv and environment secrets.
"""
from pathlib import Path
import pandas as pd

BLOCKED_FILES = {"ground_truth.csv", ".env"}

def load_table(table_name: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Safely load a dataset table for audit analysis.
    Raises PermissionError if an agent attempts to access restricted files.
    """
    path = Path(data_dir) / table_name
    normalized_name = path.name.lower()
    
    if normalized_name in BLOCKED_FILES or "ground_truth" in normalized_name:
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
