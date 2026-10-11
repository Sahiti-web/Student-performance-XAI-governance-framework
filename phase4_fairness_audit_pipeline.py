"""
Phase 4: Algorithmic Fairness Audit Pipeline & Ethical Governance Reporting
=============================================================================
Audits the champion predictive model for algorithmic bias across protected
demographic and socioeconomic attributes:
  - Gender (`genero_F`, `genero_M`)
  - Socioeconomic status / Family Gross Income (`rendabruta`)

Computes:
  1. Statistical Parity Difference (Demographic Parity)
  2. Disparate Impact Ratio (80% / Four-Fifths Rule)
  3. Equal Opportunity Difference (True Positive Rate Parity)
  4. Average Odds Difference (Equalized Odds)
"""

import sys
import io
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Console encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

BUNDLE_PATH = "phase3_bundle.pkl"
OUTPUT_DIR = "phase4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_group_metrics(y_true, y_pred, mask):
    """Compute TPR, FPR, and Positive Prediction Rate for a subpopulation."""
    y_t = y_true[mask]
    y_p = y_pred[mask]
    n = len(y_t)
    if n == 0:
        return {"n": 0, "selection_rate": 0, "tpr": 0, "fpr": 0}
    
    pos_pred = (y_p == 1).sum()
    selection_rate = pos_pred / n

    actual_pos = (y_t == 1).sum()
    actual_neg = (y_t == 0).sum()

    tp = ((y_t == 1) & (y_p == 1)).sum()
    fp = ((y_t == 0) & (y_p == 1)).sum()

    tpr = tp / actual_pos if actual_pos > 0 else 0.0
    fpr = fp / actual_neg if actual_neg > 0 else 0.0

    return {
        "n": n,
        "selection_rate": selection_rate,
        "tpr": tpr,
        "fpr": fpr,
        "actual_pos": actual_pos,
        "tp": tp
    }

def run_phase4_fairness_audit():
    print("=" * 72)
    print("PHASE 4: ALGORITHMIC FAIRNESS AUDIT PIPELINE")
    print("=" * 72)

    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Missing bundle: {BUNDLE_PATH}")

    print(f"\n[1/4] Loading champion model bundle from '{BUNDLE_PATH}'...")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    print(f"  Predicting outcomes across held-out cohort of {len(y_test)} students...")
    y_pred = model.predict(X_test)

    # ── 2. AUDIT DEFINITIONS ──────────────────────────────────────────────
    audits = []

    # A. Gender Audit
    if "genero_F" in feature_names and "genero_M" in feature_names:
        idx_f = feature_names.index("genero_F")
        idx_m = feature_names.index("genero_M")

        mask_female = X_test[:, idx_f] == 1
        mask_male = X_test[:, idx_m] == 1

        met_female = compute_group_metrics(y_test, y_pred, mask_female)
        met_male = compute_group_metrics(y_test, y_pred, mask_male)

        # Baseline: Male (privileged), Protected: Female (unprivileged)
        disp_impact_gender = met_female["selection_rate"] / met_male["selection_rate"] if met_male["selection_rate"] > 0 else 1.0
        stat_parity_gender = met_female["selection_rate"] - met_male["selection_rate"]
        eq_opp_gender = met_female["tpr"] - met_male["tpr"]
        eq_odds_gender = 0.5 * ((met_female["fpr"] - met_male["fpr"]) + (met_female["tpr"] - met_male["tpr"]))

        audits.append({
            "Attribute": "Gender",
            "Privileged_Group": "Male",
            "Unprivileged_Group": "Female",
            "Privileged_Count": met_male["n"],
            "Unprivileged_Count": met_female["n"],
            "Privileged_Recall_TPR": round(met_male["tpr"], 3),
            "Unprivileged_Recall_TPR": round(met_female["tpr"], 3),
            "Disparate_Impact_Ratio": round(disp_impact_gender, 3),
            "Statistical_Parity_Diff": round(stat_parity_gender, 3),
            "Equal_Opportunity_Diff": round(eq_opp_gender, 3),
            "Equalized_Odds_Diff": round(eq_odds_gender, 3),
            "Status_80pct_Rule": "FAIR (Passed)" if 0.80 <= disp_impact_gender <= 1.25 else "ADVERSE IMPACT"
        })

    # B. Socioeconomic Income Audit
    if "rendabruta" in feature_names:
        idx_income = feature_names.index("rendabruta")
        median_income = np.median(X_test[:, idx_income])

        mask_low_inc = X_test[:, idx_income] <= median_income
        mask_high_inc = X_test[:, idx_income] > median_income

        met_low = compute_group_metrics(y_test, y_pred, mask_low_inc)
        met_high = compute_group_metrics(y_test, y_pred, mask_high_inc)

        disp_impact_inc = met_low["selection_rate"] / met_high["selection_rate"] if met_high["selection_rate"] > 0 else 1.0
        stat_parity_inc = met_low["selection_rate"] - met_high["selection_rate"]
        eq_opp_inc = met_low["tpr"] - met_high["tpr"]
        eq_odds_inc = 0.5 * ((met_low["fpr"] - met_high["fpr"]) + (met_low["tpr"] - met_high["tpr"]))

        audits.append({
            "Attribute": "Family Income",
            "Privileged_Group": "Above Median",
            "Unprivileged_Group": "At/Below Median",
            "Privileged_Count": met_high["n"],
            "Unprivileged_Count": met_low["n"],
            "Privileged_Recall_TPR": round(met_high["tpr"], 3),
            "Unprivileged_Recall_TPR": round(met_low["tpr"], 3),
            "Disparate_Impact_Ratio": round(disp_impact_inc, 3),
            "Statistical_Parity_Diff": round(stat_parity_inc, 3),
            "Equal_Opportunity_Diff": round(eq_opp_inc, 3),
            "Equalized_Odds_Diff": round(eq_odds_inc, 3),
            "Status_80pct_Rule": "FAIR (Passed)" if 0.80 <= disp_impact_inc <= 1.25 else "MONITOR (Borderline)"
        })

    df_fairness = pd.DataFrame(audits)
    
    print("\n[3/4] Fairness Audit Results:")
    print(df_fairness.to_string(index=False))

    # Save CSV Audit Report
    csv_path = os.path.join(OUTPUT_DIR, "fairness_metrics_summary.csv")
    df_fairness.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved: {csv_path}")

    # ── 4. VISUAL FAIRNESS PARITY PLOT ───────────────────────────────────
    print("\n[4/4] Generating Visual Fairness Parity Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Disparate Impact Plot
    categories = df_fairness["Attribute"].tolist()
    di_scores = df_fairness["Disparate_Impact_Ratio"].tolist()
    bars = axes[0].bar(categories, di_scores, color=["#3b82f6", "#10b981"], width=0.45)
    axes[0].axhline(0.80, color="#ef4444", linestyle="--", linewidth=1.5, label="80% Fair Minimum (0.80)")
    axes[0].axhline(1.25, color="#ef4444", linestyle="--", linewidth=1.5, label="125% Fair Maximum (1.25)")
    axes[0].set_ylim(0, 1.5)
    axes[0].set_ylabel("Disparate Impact Ratio")
    axes[0].set_title("Disparate Impact Ratio (Four-Fifths Standard)", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    for bar, val in zip(bars, di_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.3f}", ha="center", fontweight="bold")

    # Equal Opportunity Parity Plot
    priv_recalls = df_fairness["Privileged_Recall_TPR"].tolist()
    unpriv_recalls = df_fairness["Unprivileged_Recall_TPR"].tolist()
    x = np.arange(len(categories))
    width = 0.25

    axes[1].bar(x - width/2, priv_recalls, width, label="Privileged Group", color="#6366f1")
    axes[1].bar(x + width/2, unpriv_recalls, width, label="Unprivileged (Protected)", color="#ec4899")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("Equal Opportunity Parity (Recall per Group)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "01_fairness_parity_audit.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {chart_path}")

    print("\n" + "=" * 72)
    print("✅ PHASE 4 ALGORITHMIC FAIRNESS AUDIT COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    run_phase4_fairness_audit()
