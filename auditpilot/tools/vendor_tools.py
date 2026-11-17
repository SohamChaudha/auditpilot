"""
auditpilot/tools/vendor_tools.py - Forensic vendor analysis and KYC audit tools.
Implements detection for vendor-employee shared bank accounts (conflict of interest)
and ghost/shell vendor identification.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from auditpilot.tools.data_loader import load_table


def _ensure_df(input_val: pd.DataFrame | str) -> pd.DataFrame:
    """Helper to ensure input is a DataFrame."""
    if isinstance(input_val, pd.DataFrame):
        return input_val.copy()
    return load_table(input_val)


def vendor_bank_overlap(
    ledger: pd.DataFrame | str = "data/ledger.csv",
    vendors: pd.DataFrame | str = "data/vendors.csv",
    employees: pd.DataFrame | str = "data/employees.csv",
) -> dict[str, Any]:
    """
    Identify conflict of interest and kickback risks where vendors share bank accounts with employees.
    
    Parameters:
        ledger: Ledger DataFrame or file path string.
        vendors: Vendors DataFrame or file path string.
        employees: Employees DataFrame or file path string.
        
    Returns:
        Standard audit tool dictionary.
    """
    df_ledger = _ensure_df(ledger)
    df_vendors = _ensure_df(vendors)
    df_employees = _ensure_df(employees)

    # Find matching bank accounts
    vendor_accounts = set(df_vendors["bank_account"].dropna())
    employee_accounts = set(df_employees["bank_account"].dropna())
    overlap_accounts = vendor_accounts.intersection(employee_accounts)

    implicated_vendors = df_vendors[df_vendors["bank_account"].isin(overlap_accounts)]["vendor_id"].tolist()
    implicated_employees = df_employees[df_employees["bank_account"].isin(overlap_accounts)]["employee_id"].tolist()

    # Identify all ledger disbursements to conflicted vendors
    conflicted_txns = df_ledger[df_ledger["vendor_id"].isin(implicated_vendors)]
    flagged_ids = [str(eid) for eid in conflicted_txns["entry_id"]]
    total_amount = round(float(conflicted_txns["amount_inr"].sum()), 2)

    notes = [
        f"Identified {len(overlap_accounts)} shared bank account(s) between registered vendors and company employees.",
        f"Implicated Vendors: {implicated_vendors}. Implicated Employees: {implicated_employees}.",
        f"Flagged {len(flagged_ids)} payment disbursements totaling ₹{total_amount:,.2f}."
    ]

    return {
        "test": "vendor_bank_overlap",
        "flagged_ids": flagged_ids,
        "stats": {
            "shared_accounts_count": len(overlap_accounts),
            "implicated_vendors": implicated_vendors,
            "implicated_employees": implicated_employees,
            "total_flagged_transactions": len(flagged_ids),
            "total_amount_inr": total_amount,
        },
        "notes": notes,
    }


def find_ghost_vendors(
    ledger: pd.DataFrame | str = "data/ledger.csv",
    vendors: pd.DataFrame | str = "data/vendors.csv",
) -> dict[str, Any]:
    """
    Detect shell/ghost vendors exhibiting suspicious registration or onboarding patterns.
    Checks payment dates against vendor creation dates and detects dummy GSTIN tax identifiers.
    
    Parameters:
        ledger: Ledger DataFrame or file path string.
        vendors: Vendors DataFrame or file path string.
        
    Returns:
        Standard audit tool dictionary.
    """
    df_ledger = _ensure_df(ledger)
    df_vendors = _ensure_df(vendors)

    merged = df_ledger.merge(
        df_vendors[["vendor_id", "vendor_name", "created_date", "gstin"]],
        on="vendor_id",
        how="inner",
    )

    # Chronology violation: payment voucher date is prior to vendor creation date
    chronology_mask = merged["date"] < merged["created_date"]
    
    # Dummy GSTIN indicator: state code 99 or dummy PAN prefix
    dummy_gstin_mask = merged["gstin"].astype(str).str.startswith("99")

    ghost_txns = merged[chronology_mask | dummy_gstin_mask]
    flagged_ids = [str(eid) for eid in ghost_txns["entry_id"]]
    ghost_vendors_list = list(ghost_txns["vendor_id"].unique())
    total_amount = round(float(ghost_txns["amount_inr"].sum()), 2)

    notes = [
        f"Identified {len(ghost_vendors_list)} ghost/shell vendor(s) across {len(flagged_ids)} transactions.",
        f"Ghost Vendors: {ghost_vendors_list}.",
        f"Total disbursed to unverified/post-dated vendors: ₹{total_amount:,.2f}."
    ]

    return {
        "test": "find_ghost_vendors",
        "flagged_ids": flagged_ids,
        "stats": {
            "ghost_vendors_count": len(ghost_vendors_list),
            "total_flagged_transactions": len(flagged_ids),
            "total_amount_inr": total_amount,
            "ghost_vendor_ids": ghost_vendors_list,
        },
        "notes": notes,
    }
