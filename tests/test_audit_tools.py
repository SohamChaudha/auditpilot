"""
tests/test_audit_tools.py - Comprehensive pytest validation suite for Phase 2 audit tools.
Validates standard output contracts, data integrity, ground truth isolation,
recall >= 90% across all 8 seeded anomaly types, zero false positives,
Benford's Law analysis, financial ratios, and edge-case handling.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from auditpilot.tools import (
    load_table,
    load_ledger,
    list_accessible_tables,
    check_trial_balance,
    benford_test,
    ratio_table,
    find_duplicates,
    find_split_payments,
    find_weekend_round,
    check_approval_limits,
    detect_prompt_injection,
    vendor_bank_overlap,
    find_ghost_vendors,
)

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ground_truth_df():
    """Load ground_truth.csv exclusively inside pytest for benchmark evaluation."""
    gt_path = DATA_DIR / "ground_truth.csv"
    assert gt_path.exists(), "data/ground_truth.csv must exist for evaluation."
    return pd.read_csv(gt_path)


@pytest.fixture(scope="session")
def loaded_ledger_data():
    """Pre-load ledger using load_ledger tool."""
    res = load_ledger(DATA_DIR / "ledger.csv")
    assert res["test"] == "load_ledger"
    assert "data" in res
    return res["data"]


# ---------------------------------------------------------------------------
# 1. Standard Output Contract & Structure
# ---------------------------------------------------------------------------

def test_standard_output_format_all_tools(loaded_ledger_data):
    """Verify that every tool adheres to the standard output format dictionary."""
    tools_under_test = [
        lambda: load_ledger(DATA_DIR / "ledger.csv"),
        lambda: check_trial_balance(DATA_DIR / "trial_balance.csv", loaded_ledger_data),
        lambda: benford_test(loaded_ledger_data),
        lambda: find_duplicates(loaded_ledger_data),
        lambda: find_split_payments(loaded_ledger_data),
        lambda: find_weekend_round(loaded_ledger_data),
        lambda: check_approval_limits(loaded_ledger_data, DATA_DIR / "employees.csv"),
        lambda: vendor_bank_overlap(loaded_ledger_data, DATA_DIR / "vendors.csv", DATA_DIR / "employees.csv"),
        lambda: find_ghost_vendors(loaded_ledger_data, DATA_DIR / "vendors.csv"),
        lambda: detect_prompt_injection(loaded_ledger_data),
        lambda: ratio_table(DATA_DIR / "financials.csv"),
    ]

    for tool_call in tools_under_test:
        result = tool_call()
        assert isinstance(result, dict), f"Tool {tool_call} did not return a dict"
        assert "test" in result, "Missing 'test' key"
        assert isinstance(result["test"], str), "'test' must be a str"
        assert "flagged_ids" in result, "Missing 'flagged_ids' key"
        assert isinstance(result["flagged_ids"], list), "'flagged_ids' must be a list"
        assert "stats" in result, "Missing 'stats' key"
        assert isinstance(result["stats"], dict), "'stats' must be a dict"
        assert "notes" in result, "Missing 'notes' key"
        assert isinstance(result["notes"], list), "'notes' must be a list"


def test_load_ledger_data_contract():
    """Verify load_ledger specifically provides the 'data' DataFrame field and validates schema."""
    res = load_ledger(DATA_DIR / "ledger.csv")
    assert "data" in res
    assert isinstance(res["data"], pd.DataFrame)
    assert len(res["data"]) == 6000
    assert res["stats"]["total_records"] == 6000
    assert res["stats"]["valid_records"] == 6000
    assert res["stats"]["invalid_records"] == 0
    assert len(res["flagged_ids"]) == 0


# ---------------------------------------------------------------------------
# 2. Ground Truth Technical Isolation
# ---------------------------------------------------------------------------

def test_ground_truth_isolation_enforced():
    """Verify that audit tools and data loaders strictly reject access to ground_truth.csv."""
    # Direct load_table rejection
    with pytest.raises(PermissionError):
        load_table("ground_truth.csv", data_dir="data")

    # load_ledger rejection
    with pytest.raises(PermissionError):
        load_ledger(DATA_DIR / "ground_truth.csv")

    # Case-insensitive and sub-path rejection
    with pytest.raises(PermissionError):
        load_table("data/GROUND_TRUTH.CSV")

    # Accessible table listing omits ground truth
    accessible = list_accessible_tables("data")
    assert "ground_truth.csv" not in accessible


# ---------------------------------------------------------------------------
# 3. Anomaly Recall Evaluation (>= 90% Required for Each Seeded Type)
# ---------------------------------------------------------------------------

def test_recall_duplicate_payments(loaded_ledger_data, ground_truth_df):
    """
    Evaluate duplicate payments detection:
    Recall = True Positives / (True Positives + False Negatives) >= 90%.
    """
    res = find_duplicates(loaded_ledger_data, max_days=2)
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "duplicate_payment"]["entry_id"])
    assert len(expected) == 20

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Duplicate payments recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for duplicate payments, got {recall:.2%}"

    # Zero false positives
    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


def test_recall_split_payments(loaded_ledger_data, ground_truth_df):
    """
    Evaluate split payments detection across 3 consecutive days:
    Recall >= 90%.
    """
    res = find_split_payments(loaded_ledger_data, threshold_limit=50000.0, min_amount=45000.0, window_days=3)
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "split_payment"]["entry_id"])
    assert len(expected) == 18

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Split payments recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for split payments, got {recall:.2%}"

    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


def test_recall_ghost_vendors(loaded_ledger_data, ground_truth_df):
    """
    Evaluate ghost vendor detection:
    Recall >= 90%.
    """
    res = find_ghost_vendors(loaded_ledger_data, DATA_DIR / "vendors.csv")
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "ghost_vendor"]["entry_id"])
    assert len(expected) == 5

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Ghost vendor recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for ghost vendors, got {recall:.2%}"

    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


def test_recall_shared_bank_accounts(loaded_ledger_data, ground_truth_df):
    """
    Evaluate shared bank account conflict detection:
    Recall >= 90%.
    """
    res = vendor_bank_overlap(loaded_ledger_data, DATA_DIR / "vendors.csv", DATA_DIR / "employees.csv")
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "shared_bank_account"]["entry_id"])
    assert len(expected) == 6

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Shared bank accounts recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for shared bank accounts, got {recall:.2%}"

    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


def test_recall_self_approved(loaded_ledger_data, ground_truth_df):
    """
    Evaluate self-approved transactions detection:
    Recall >= 90%.
    """
    res = check_approval_limits(loaded_ledger_data, DATA_DIR / "employees.csv")
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "self_approved"]["entry_id"])
    assert len(expected) == 8

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Self-approved recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for self-approved transactions, got {recall:.2%}"

    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


def test_recall_sunday_round_amounts(loaded_ledger_data, ground_truth_df):
    """
    Evaluate Sunday round amount transactions detection:
    Recall >= 90%.
    """
    res = find_weekend_round(loaded_ledger_data, round_modulo=10000)
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "sunday_round_amount"]["entry_id"])
    assert len(expected) == 12

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Sunday round recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for Sunday round transactions, got {recall:.2%}"

    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


def test_recall_trial_balance_gap(loaded_ledger_data, ground_truth_df):
    """
    Evaluate trial balance gap detection and ledger reconciliation:
    Recall >= 90%.
    """
    res = check_trial_balance(DATA_DIR / "trial_balance.csv", loaded_ledger_data)
    stats = res["stats"]

    assert not stats["is_balanced"], "Trial balance should be flagged as unbalanced"
    assert stats["imbalance_inr"] == 100000.0, f"Expected imbalance of ₹100,000, got ₹{stats['imbalance_inr']}"
    assert "1010" in stats["discrepant_accounts"], "Discrepant account 1010 must be identified"


def test_recall_prompt_injections(loaded_ledger_data, ground_truth_df):
    """
    Evaluate prompt injection detection:
    Recall >= 90%.
    """
    res = detect_prompt_injection(loaded_ledger_data)
    flagged = set(res["flagged_ids"])

    expected = set(ground_truth_df[ground_truth_df["anomaly_type"] == "prompt_injection"]["entry_id"])
    assert len(expected) == 4

    tp = len(flagged.intersection(expected))
    fn = len(expected - flagged)
    recall = tp / (tp + fn)

    assert recall >= 0.90, f"Prompt injection recall ({recall:.2%}) is below 90%"
    assert recall == 1.0, f"Expected 100% recall for prompt injections, got {recall:.2%}"

    fp = len(flagged - expected)
    assert fp == 0, f"Detected unexpected false positives: {flagged - expected}"


# ---------------------------------------------------------------------------
# 4. Supplementary Analytical Tools (Benford & Ratios)
# ---------------------------------------------------------------------------

def test_benford_analysis(loaded_ledger_data):
    """Verify Benford's Law distribution analysis and statistical properties."""
    res = benford_test(loaded_ledger_data)
    stats = res["stats"]

    assert stats["total_transactions"] == 6000
    assert len(stats["observed_distribution"]) == 9
    assert len(stats["expected_distribution"]) == 9

    # Check that probabilities sum to approximately 1.0
    obs_sum = sum(stats["observed_distribution"].values())
    assert 0.99 <= obs_sum <= 1.01

    # Check Chi-square statistic is positive and MAD is computed
    assert stats["chi_square_stat"] > 0
    assert stats["mad"] > 0
    assert stats["conformity"] in ["Close conformity", "Acceptable conformity", "Marginally acceptable conformity", "Nonconformity"]


def test_ratio_table_formulas():
    """Verify financial ratio table formulas and mathematical accuracy."""
    res = ratio_table(DATA_DIR / "financials.csv")
    stats = res["stats"]
    metrics = stats["monthly_metrics"]

    assert stats["months_analyzed"] == 12
    assert len(metrics) == 12

    fin_df = pd.read_csv(DATA_DIR / "financials.csv")

    for i, m in enumerate(metrics):
        row = fin_df.iloc[i]
        rev = float(row["revenue_inr"])
        gp = float(row["gross_profit_inr"])
        op = float(row["operating_income_inr"])
        net = float(row["net_income_inr"])

        expected_gm = round((gp / rev) * 100, 2)
        expected_om = round((op / rev) * 100, 2)
        expected_nm = round((net / rev) * 100, 2)

        assert abs(m["gross_margin_pct"] - expected_gm) < 0.01
        assert abs(m["operating_margin_pct"] - expected_om) < 0.01
        assert abs(m["net_margin_pct"] - expected_nm) < 0.01


# ---------------------------------------------------------------------------
# 5. Missing / Invalid Data Handling
# ---------------------------------------------------------------------------

def test_missing_column_error_handling(tmp_path):
    """Verify that load_ledger raises ValueError on missing required columns."""
    bad_csv = tmp_path / "bad_ledger.csv"
    bad_df = pd.DataFrame({"entry_id": ["TXN-001"], "amount_inr": [100.0]})
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        load_ledger(bad_csv)
    assert "missing required columns" in str(exc_info.value).lower()


def test_empty_dataframe_handling():
    """Verify tools handle empty DataFrames gracefully without unhandled exceptions."""
    empty_df = pd.DataFrame(columns=["entry_id", "date", "posted_by", "approved_by", "account_code", "amount_inr", "description", "vendor_id"])
    
    # Empty ledger should return 0 flagged and empty stats cleanly
    dup_res = find_duplicates(empty_df)
    assert dup_res["flagged_ids"] == []
    assert dup_res["stats"]["total_flagged_transactions"] == 0

    split_res = find_split_payments(empty_df)
    assert split_res["flagged_ids"] == []

    weekend_res = find_weekend_round(empty_df)
    assert weekend_res["flagged_ids"] == []
