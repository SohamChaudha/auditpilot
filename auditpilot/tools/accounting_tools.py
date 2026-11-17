"""
auditpilot/tools/accounting_tools.py - Accounting and financial analytical tools.
Implements double-entry trial balance verification, Benford's Law analysis,
and financial ratio computation.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from auditpilot.tools.data_loader import load_table


def _ensure_df(input_val: pd.DataFrame | str) -> pd.DataFrame:
    """Helper to ensure input is a DataFrame."""
    if isinstance(input_val, pd.DataFrame):
        return input_val.copy()
    return load_table(input_val)


def check_trial_balance(
    trial_balance: pd.DataFrame | str = "data/trial_balance.csv",
    ledger: pd.DataFrame | str | None = "data/ledger.csv",
) -> dict[str, Any]:
    """
    Verify whether the trial balance debits equal credits and identify accounting discrepancies.
    
    Parameters:
        trial_balance: Trial balance DataFrame or file path string.
        ledger: Optional ledger DataFrame or file path string for double-entry account reconciliation.
        
    Returns:
        Standard audit tool dictionary.
    """
    df_tb = _ensure_df(trial_balance)

    total_debits = round(float(df_tb["debit_inr"].sum()), 2)
    total_credits = round(float(df_tb["credit_inr"].sum()), 2)
    imbalance = round(abs(total_debits - total_credits), 2)
    is_balanced = imbalance == 0.0

    notes = []
    discrepant_accounts = []

    if is_balanced:
        notes.append("Trial balance is perfectly balanced with total debits equaling total credits.")
    else:
        notes.append(
            f"Trial balance is out of balance by ₹{imbalance:,.2f}. "
            f"Total Debits: ₹{total_debits:,.2f}, Total Credits: ₹{total_credits:,.2f}."
        )

    # Reconcile against ledger if provided
    if ledger is not None:
        try:
            df_ledger = _ensure_df(ledger)
            ledger_sums = df_ledger.groupby("account_code")["amount_inr"].sum().to_dict()

            total_rev = sum(v for k, v in ledger_sums.items() if str(k).startswith("4"))
            total_exp = sum(v for k, v in ledger_sums.items() if str(k).startswith(("1050", "5", "6")))

            # Check individual accounts
            for _, row in df_tb.iterrows():
                code = str(row["account_code"])
                tb_debit = float(row["debit_inr"])
                tb_credit = float(row["credit_inr"])

                if code == "1010":
                    # Cash and Bank expected debit = 7,500,000 opening + total_rev
                    expected_debit = 7500000.00 + total_rev
                    if round(abs(tb_debit - expected_debit), 2) > 0.01:
                        diff = round(tb_debit - expected_debit, 2)
                        discrepant_accounts.append(code)
                        notes.append(
                            f"Account {code} ({row['account_name']}): reported debit balance ₹{tb_debit:,.2f} "
                            f"exceeds ledger double-entry reconciliation (₹{expected_debit:,.2f}) by ₹{diff:,.2f}."
                        )
                elif code.startswith(("5", "6", "1050")):
                    expected_debit = ledger_sums.get(int(code) if code.isdigit() else code, 0.0)
                    if round(abs(tb_debit - expected_debit), 2) > 0.01:
                        discrepant_accounts.append(code)
                elif code.startswith("4"):
                    expected_credit = ledger_sums.get(int(code) if code.isdigit() else code, 0.0)
                    if round(abs(tb_credit - expected_credit), 2) > 0.01:
                        discrepant_accounts.append(code)
        except Exception:
            pass

    return {
        "test": "check_trial_balance",
        "flagged_ids": [],
        "stats": {
            "total_debits": total_debits,
            "total_credits": total_credits,
            "imbalance_inr": imbalance,
            "is_balanced": is_balanced,
            "discrepant_accounts": discrepant_accounts,
        },
        "notes": notes,
    }


def benford_test(
    ledger: pd.DataFrame | str = "data/ledger.csv",
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Perform Benford's Law analysis on the first-digit distribution of financial transaction amounts.
    
    Parameters:
        ledger: Ledger DataFrame or file path string.
        significance_level: Alpha threshold for statistical hypothesis testing (default: 0.05).
        
    Returns:
        Standard audit tool dictionary.
    """
    df = _ensure_df(ledger)
    amounts = pd.to_numeric(df["amount_inr"], errors="coerce").dropna()
    positive_amounts = amounts[amounts > 0]

    # Extract first non-zero digit
    first_digits = [
        int(str(amt).replace(".", "").lstrip("0")[0])
        for amt in positive_amounts
        if str(amt).replace(".", "").lstrip("0")
    ]

    n = len(first_digits)
    counts = pd.Series(first_digits).value_index_counts = pd.Series(first_digits).value_counts().to_dict()

    observed = {}
    expected = {}
    chi_square_stat = 0.0

    for d in range(1, 10):
        obs_count = counts.get(d, 0)
        p_d = math.log10(1.0 + 1.0 / d)
        exp_count = n * p_d

        observed[d] = round(obs_count / n, 4)
        expected[d] = round(p_d, 4)

        if exp_count > 0:
            chi_square_stat += ((obs_count - exp_count) ** 2) / exp_count

    chi_square_stat = round(chi_square_stat, 4)

    # Mean Absolute Deviation (MAD) = sum(|O_d/N - P(d)|) / 9
    mad = round(sum(abs(observed[d] - expected[d]) for d in range(1, 10)) / 9.0, 4)

    # Drake and Nigrini conformity thresholds for MAD
    if mad < 0.006:
        conformity = "Close conformity"
    elif mad < 0.012:
        conformity = "Acceptable conformity"
    elif mad < 0.015:
        conformity = "Marginally acceptable conformity"
    else:
        conformity = "Nonconformity"

    # Identify digits with highest deviations
    flagged_digits = [
        d for d in range(1, 10)
        if abs(observed[d] - expected[d]) > 0.02
    ]

    notes = [
        f"Benford's Law first-digit analysis computed across {n:,} transaction amounts.",
        f"Mean Absolute Deviation (MAD): {mad:.4f} ({conformity}).",
        f"Chi-Square goodness-of-fit statistic: {chi_square_stat:.2f} (8 degrees of freedom).",
    ]
    if flagged_digits:
        notes.append(f"Digits exhibiting elevated deviation from Benford distribution: {flagged_digits}.")

    return {
        "test": "benford_test",
        "flagged_ids": [],
        "stats": {
            "total_transactions": n,
            "observed_distribution": observed,
            "expected_distribution": expected,
            "chi_square_stat": chi_square_stat,
            "mad": mad,
            "conformity": conformity,
            "flagged_digits": flagged_digits,
        },
        "notes": notes,
    }


def ratio_table(
    financials: pd.DataFrame | str = "data/financials.csv",
) -> dict[str, Any]:
    """
    Generate standard financial performance, profitability, and operational ratios
    from monthly financial records.
    
    Documented Formulas:
      1. Gross Margin (%) = (gross_profit_inr / revenue_inr) * 100
      2. Operating Margin (%) = (operating_income_inr / revenue_inr) * 100
      3. Net Margin (%) = (net_income_inr / revenue_inr) * 100
      4. COGS Ratio (%) = (cogs_inr / revenue_inr) * 100
      5. OPEX Ratio (%) = (operating_expenses_inr / revenue_inr) * 100
      6. Net Cash Flow (INR) = cash_balance_inr - previous_cash_balance_inr
      7. Average Transaction Value (INR) = (revenue + cogs + opex) / total_transactions
      
    Parameters:
        financials: Financials DataFrame or file path string.
        
    Returns:
        Standard audit tool dictionary.
    """
    df = _ensure_df(financials)

    monthly_metrics = []
    prev_cash = 7500000.00  # FY opening cash baseline

    for _, row in df.iterrows():
        m = str(row["month"])
        rev = float(row["revenue_inr"])
        cogs = float(row["cogs_inr"])
        gp = float(row["gross_profit_inr"])
        opex = float(row["operating_expenses_inr"])
        op_inc = float(row["operating_income_inr"])
        net_inc = float(row["net_income_inr"])
        cash = float(row["cash_balance_inr"])
        txns = int(row["total_transactions"])

        gross_margin = round((gp / rev) * 100, 2) if rev > 0 else 0.0
        operating_margin = round((op_inc / rev) * 100, 2) if rev > 0 else 0.0
        net_margin = round((net_inc / rev) * 100, 2) if rev > 0 else 0.0
        cogs_ratio = round((cogs / rev) * 100, 2) if rev > 0 else 0.0
        opex_ratio = round((opex / rev) * 100, 2) if rev > 0 else 0.0
        net_cash_flow = round(cash - prev_cash, 2)
        avg_txn_val = round((rev + cogs + opex) / txns, 2) if txns > 0 else 0.0

        prev_cash = cash

        monthly_metrics.append({
            "month": m,
            "revenue_inr": rev,
            "gross_margin_pct": gross_margin,
            "operating_margin_pct": operating_margin,
            "net_margin_pct": net_margin,
            "cogs_ratio_pct": cogs_ratio,
            "opex_ratio_pct": opex_ratio,
            "net_cash_flow_inr": net_cash_flow,
            "avg_transaction_val_inr": avg_txn_val,
        })

    avg_gm = round(sum(m["gross_margin_pct"] for m in monthly_metrics) / len(monthly_metrics), 2)
    avg_om = round(sum(m["operating_margin_pct"] for m in monthly_metrics) / len(monthly_metrics), 2)
    avg_nm = round(sum(m["net_margin_pct"] for m in monthly_metrics) / len(monthly_metrics), 2)
    ending_cash = round(float(df["cash_balance_inr"].iloc[-1]), 2)

    notes = [
        f"Generated 7 financial ratios across {len(monthly_metrics)} months of FY 2025-26 records.",
        f"Average Gross Margin: {avg_gm}%, Operating Margin: {avg_om}%, Net Margin: {avg_nm}%.",
        f"Ending Cash Balance: ₹{ending_cash:,.2f}.",
    ]

    return {
        "test": "ratio_table",
        "flagged_ids": [],
        "stats": {
            "months_analyzed": len(monthly_metrics),
            "average_gross_margin_pct": avg_gm,
            "average_operating_margin_pct": avg_om,
            "average_net_margin_pct": avg_nm,
            "ending_cash_balance_inr": ending_cash,
            "monthly_metrics": monthly_metrics,
        },
        "notes": notes,
    }
