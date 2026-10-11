"""
Phase 4: Production Algorithmic Fairness Audit Pipeline (with Race Equity Audit)
================================================================================
Comprehensive Institutional Audit:
  1. Gender: Male (privileged) vs. Female (unprivileged)
  2. Race / Ethnicity: White/Branca (privileged) vs. Mixed/Parda & Black/Preta (unprivileged)
  3. Socioeconomic Income: Above Median (privileged) vs. At/Below Median (unprivileged)
  4. Age Cohort: Traditional <=20 (privileged) vs. Adult >=21 (unprivileged)
  5. Digital Access: >=1 Laptop (privileged) vs. 0 Laptops (unprivileged)

Computes:
  - Disparate Impact Ratio (80% / Four-Fifths Rule)
  - Statistical Parity Difference (SPD)
  - Equal Opportunity Difference (EOD / Recall Parity)
  - Average Odds Difference (Equalized Odds)
  - Decision Threshold Recalibration Matrix
  - 300 DPI Publication-grade Audit Visualizations
"""

import sys
import io
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

BUNDLE_PATH = "phase3_bundle.pkl"
RACE_PATH = "phase4_outputs/test_cohort_races.npy"
OUTPUT_DIR = "phase4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_group_metrics(y_true, y_pred, mask):
    y_t = y_true[mask]
    y_p = y_pred[mask]
    n = len(y_t)
    if n == 0:
        return {"n": 0, "pos_rate": 0.0, "tpr": 0.0, "fpr": 0.0, "precision": 0.0, "accuracy": 0.0}
    
    pos_pred = int((y_p == 1).sum())
    selection_rate = float(pos_pred / n)
    actual_pos = int((y_t == 1).sum())
    actual_neg = int((y_t == 0).sum())

    tp = int(((y_t == 1) & (y_p == 1)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    tn = int(((y_t == 0) & (y_p == 0)).sum())
    fn = int(((y_t == 1) & (y_p == 0)).sum())

    tpr = float(tp / actual_pos) if actual_pos > 0 else 0.0
    fpr = float(fp / actual_neg) if actual_neg > 0 else 0.0
    precision = float(tp / pos_pred) if pos_pred > 0 else 0.0
    accuracy = float((tp + tn) / n) if n > 0 else 0.0

    return {
        "n": n,
        "actual_pos": actual_pos,
        "actual_neg": actual_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "selection_rate": selection_rate,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "accuracy": accuracy
    }

def run_phase4_audit():
    print("=" * 80)
    print("⚖️ PHASE 4: EXPANDED ALGORITHMIC FAIRNESS AUDIT (INCLUDING RACE EQUITY)")
    print("=" * 80)

    # 1. Load Model Bundle
    print(f"\n[1/6] Loading Champion Bundle '{BUNDLE_PATH}'...")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    probs = model.predict_proba(X_test)[:, 1]
    y_pred_default = (probs >= 0.50).astype(int)

    # 2. Load Race Demographics
    print(f"[2/6] Loading Test Cohort Race Demographics '{RACE_PATH}'...")
    if not os.path.exists(RACE_PATH):
        raise FileNotFoundError("Run race mapping script first!")
    races = np.load(RACE_PATH, allow_pickle=True)
    print(f"  ✓ Total Test Cohort: {len(races)} student records aligned")

    # 3. Configure All Protected Slices
    print("\n[3/6] Configuring Protected Demographic & Socioeconomic Slices...")
    slices = []

    # Slice 1: Gender (Male = Privileged, Female = Unprivileged)
    idx_f = feature_names.index("genero_F")
    idx_m = feature_names.index("genero_M")
    mask_male = (X_test[:, idx_m] == 1)
    mask_female = (X_test[:, idx_f] == 1)
    slices.append({
        "dimension": "Gender",
        "priv_name": "Male",
        "unpriv_name": "Female",
        "priv_mask": mask_male,
        "unpriv_mask": mask_female
    })

    # Slice 2: Race / Ethnicity (Branca = Privileged, Parda/Preta = Unprivileged)
    mask_white = (races == "Branca")
    mask_minority = np.isin(races, ["Parda", "Preta", "Indígena", "Amarela"])
    slices.append({
        "dimension": "Race / Ethnicity",
        "priv_name": "White (Branca)",
        "unpriv_name": "Brown/Black/Indigenous (Parda/Preta)",
        "priv_mask": mask_white,
        "unpriv_mask": mask_minority
    })

    # Slice 3: Family Income (Above Median = Privileged, Below/At Median = Unprivileged)
    idx_inc = feature_names.index("rendabruta")
    med_inc = float(np.median(X_test[:, idx_inc]))
    mask_high_inc = (X_test[:, idx_inc] > med_inc)
    mask_low_inc = (X_test[:, idx_inc] <= med_inc)
    slices.append({
        "dimension": "Household Income",
        "priv_name": f"> Median (R$ {med_inc:.0f})",
        "unpriv_name": f"<= Median (R$ {med_inc:.0f})",
        "priv_mask": mask_high_inc,
        "unpriv_mask": mask_low_inc
    })

    # Slice 4: Student Age (Traditional <=20 vs. Adult/Non-traditional >=21)
    idx_age = feature_names.index("idade")
    mask_trad_age = (X_test[:, idx_age] <= 20)
    mask_adult_age = (X_test[:, idx_age] >= 21)
    slices.append({
        "dimension": "Age Cohort",
        "priv_name": "Traditional (<= 20 yrs)",
        "unpriv_name": "Adult / Non-traditional (>= 21 yrs)",
        "priv_mask": mask_trad_age,
        "unpriv_mask": mask_adult_age
    })

    # Slice 5: Digital Access / Laptop Ownership (>=1 laptop vs. 0 laptops)
    idx_laptop = feature_names.index("quantidade_notebooks")
    mask_has_laptop = (X_test[:, idx_laptop] >= 1)
    mask_no_laptop = (X_test[:, idx_laptop] == 0)
    slices.append({
        "dimension": "Digital Access",
        "priv_name": "Owns >=1 Laptop",
        "unpriv_name": "No Laptop (0)",
        "priv_mask": mask_has_laptop,
        "unpriv_mask": mask_no_laptop
    })

    # 4. Compute Quantitative Fairness Metrics
    print("\n[4/6] Computing Disparate Impact, Statistical Parity, and Equal Opportunity...")
    audit_records = []

    for s in slices:
        m_priv = compute_group_metrics(y_test, y_pred_default, s["priv_mask"])
        m_unpriv = compute_group_metrics(y_test, y_pred_default, s["unpriv_mask"])

        di_ratio = m_unpriv["selection_rate"] / m_priv["selection_rate"] if m_priv["selection_rate"] > 0 else 1.0
        spd = m_unpriv["selection_rate"] - m_priv["selection_rate"]
        eod = m_unpriv["tpr"] - m_priv["tpr"]
        aod = 0.5 * ((m_unpriv["fpr"] - m_priv["fpr"]) + (m_unpriv["tpr"] - m_priv["tpr"]))

        # Compliance Check against 80% Rule (0.80 <= DI <= 1.25)
        if 0.80 <= di_ratio <= 1.25:
            di_status = "PASSED (Fair)"
            badge_color = "success"
        elif di_ratio < 0.80:
            di_status = "UNDER-FLAGGED (<0.80)"
            badge_color = "warning"
        else:
            di_status = "OVER-FLAGGED (>1.25)"
            badge_color = "danger"

        eod_status = "OPTIMAL (|gap| <= 5%)" if abs(eod) <= 0.05 else "DISPARATE SENSITIVITY"

        audit_records.append({
            "Dimension": s["dimension"],
            "Privileged_Group": s["priv_name"],
            "Unprivileged_Group": s["unpriv_name"],
            "Priv_Count": m_priv["n"],
            "Unpriv_Count": m_unpriv["n"],
            "Priv_Dropout_Rate_Actual": round(m_priv["actual_pos"] / m_priv["n"] * 100, 1) if m_priv["n"] > 0 else 0,
            "Unpriv_Dropout_Rate_Actual": round(m_unpriv["actual_pos"] / m_unpriv["n"] * 100, 1) if m_unpriv["n"] > 0 else 0,
            "Priv_Selection_Rate": round(m_priv["selection_rate"] * 100, 1),
            "Unpriv_Selection_Rate": round(m_unpriv["selection_rate"] * 100, 1),
            "Priv_Recall_TPR": round(m_priv["tpr"] * 100, 1),
            "Unpriv_Recall_TPR": round(m_unpriv["tpr"] * 100, 1),
            "Priv_FPR": round(m_priv["fpr"] * 100, 1),
            "Unpriv_FPR": round(m_unpriv["fpr"] * 100, 1),
            "Disparate_Impact_Ratio": round(di_ratio, 3),
            "Statistical_Parity_Diff": round(spd, 3),
            "Equal_Opportunity_Diff": round(eod, 3),
            "Average_Odds_Diff": round(aod, 3),
            "Four_Fifths_Status": di_status,
            "Equal_Opp_Status": eod_status,
            "Badge": badge_color
        })

    df_fairness = pd.DataFrame(audit_records)
    csv_path = os.path.join(OUTPUT_DIR, "fairness_metrics_summary.csv")
    df_fairness.to_csv(csv_path, index=False)
    print(f"  ✓ Saved Quantitative Audit: {csv_path}")

    # Display console table
    print("\n" + df_fairness[["Dimension", "Disparate_Impact_Ratio", "Equal_Opportunity_Diff", "Four_Fifths_Status", "Equal_Opp_Status"]].to_string(index=False))

    # 4B. Intersectional Fairness Matrix (Gender x Race x Income - 100% Real Data)
    print("\n[4B] Computing Intersectional Matrix (Gender × Race × Income)...")
    intersectional_records = []
    for g_name, g_mask in [("Female", mask_female), ("Male", mask_male)]:
        for r_name, r_mask in [("Minority", mask_minority), ("White", mask_white)]:
            for i_name, i_mask in [("Low Income", mask_low_inc), ("High Income", mask_high_inc)]:
                compound_mask = g_mask & r_mask & i_mask
                m_sub = compute_group_metrics(y_test, y_pred_default, compound_mask)
                if m_sub["n"] > 0:
                    intersectional_records.append({
                        "Subgroup": f"{g_name} | {r_name} | {i_name}",
                        "Count": m_sub["n"],
                        "Actual_Dropout_Rate": round(m_sub["actual_pos"] / m_sub["n"] * 100, 1),
                        "Flagged_Rate": round(m_sub["selection_rate"] * 100, 1),
                        "Recall_TPR": round(m_sub["tpr"] * 100, 1),
                        "Precision": round(m_sub["precision"] * 100, 1)
                    })
    df_intersectional = pd.DataFrame(intersectional_records).sort_values("Count", ascending=False)
    csv_inter_path = os.path.join(OUTPUT_DIR, "intersectional_fairness_summary.csv")
    df_intersectional.to_csv(csv_inter_path, index=False)
    print(f"  ✓ Saved Intersectional Fairness Audit: {csv_inter_path}")

    # 4C. Early-Warning Active Students vs. Post-Mortem Records Audit (ira > 0 vs ira == 0)
    print("\n[4C] Computing Early-Warning vs. Administrative Post-Mortem Audit...")
    idx_ira = feature_names.index("ira")
    mask_active = (X_test[:, idx_ira] > 0)
    mask_postmortem = (X_test[:, idx_ira] == 0)

    m_active = compute_group_metrics(y_test, y_pred_default, mask_active)
    m_postmortem = compute_group_metrics(y_test, y_pred_default, mask_postmortem)
    m_all = compute_group_metrics(y_test, y_pred_default, np.ones(len(y_test), dtype=bool))

    ew_records = [
        {
            "Cohort": "Early-Warning Active Students (GPA > 0)",
            "Count": m_active["n"],
            "Actual_Dropouts": m_active["actual_pos"],
            "Actual_Rate": round(m_active["actual_pos"] / m_active["n"] * 100, 1),
            "Recall_Sensitivity": round(m_active["tpr"] * 100, 1),
            "Precision": round(m_active["precision"] * 100, 1)
        },
        {
            "Cohort": "Administrative Post-Mortem Records (GPA == 0)",
            "Count": m_postmortem["n"],
            "Actual_Dropouts": m_postmortem["actual_pos"],
            "Actual_Rate": round(m_postmortem["actual_pos"] / m_postmortem["n"] * 100, 1),
            "Recall_Sensitivity": round(m_postmortem["tpr"] * 100, 1),
            "Precision": round(m_postmortem["precision"] * 100, 1)
        },
        {
            "Cohort": "Combined Full Test Set",
            "Count": m_all["n"],
            "Actual_Dropouts": m_all["actual_pos"],
            "Actual_Rate": round(m_all["actual_pos"] / m_all["n"] * 100, 1),
            "Recall_Sensitivity": round(m_all["tpr"] * 100, 1),
            "Precision": round(m_all["precision"] * 100, 1)
        }
    ]
    df_ew = pd.DataFrame(ew_records)
    csv_ew_path = os.path.join(OUTPUT_DIR, "early_warning_vs_postmortem_audit.csv")
    df_ew.to_csv(csv_ew_path, index=False)
    print(f"  ✓ Saved Early-Warning Audit: {csv_ew_path}")

    # 4D. Course Modality Disaggregated Audit
    print("\n[4D] Computing Course Modality Disaggregated Baseline Audit...")
    idx_mod = feature_names.index("modalidade")
    modality_labels = {
        1.0: "Integrated High School (Integrado - Traditional Teens)",
        3.0: "Subsequent Technical (Subsequente - Post-High School)",
        0.0: "Concurrent Technical (Concomitante - Dual Schooling)",
        2.0: "Youth & Adult Education (Integrado EJA - Adults)"
    }
    mod_records = []
    for code, label in modality_labels.items():
        mask_mod = (X_test[:, idx_mod] == code)
        m_mod = compute_group_metrics(y_test, y_pred_default, mask_mod)
        if m_mod["n"] > 0:
            mod_records.append({
                "Course_Modality": label,
                "Total_Students": m_mod["n"],
                "Actual_Dropout_Rate": round(m_mod["actual_pos"] / m_mod["n"] * 100, 1),
                "Model_Flagging_Rate": round(m_mod["selection_rate"] * 100, 1),
                "Model_Recall_TPR": round(m_mod["tpr"] * 100, 1),
                "Model_Precision": round(m_mod["precision"] * 100, 1)
            })
    df_mod = pd.DataFrame(mod_records).sort_values("Total_Students", ascending=False)
    csv_mod_path = os.path.join(OUTPUT_DIR, "modality_disaggregated_audit.csv")
    df_mod.to_csv(csv_mod_path, index=False)
    print(f"  ✓ Saved Course Modality Audit: {csv_mod_path}")
    print("\n[5/6] Decision Threshold Recalibration Analysis...")
    threshold_grid = np.linspace(0.30, 0.70, 9)
    threshold_results = []
    
    for th in threshold_grid:
        y_th = (probs >= th).astype(int)
        global_rec = float((y_test[y_test == 1] == y_th[y_test == 1]).mean())
        global_prec = float(((y_test == 1) & (y_th == 1)).sum() / (y_th == 1).sum()) if (y_th == 1).sum() > 0 else 0
        
        m_f = compute_group_metrics(y_test, y_th, mask_female)
        m_m = compute_group_metrics(y_test, y_th, mask_male)
        di_g = m_f["selection_rate"] / m_m["selection_rate"] if m_m["selection_rate"] > 0 else 1.0
        
        m_min = compute_group_metrics(y_test, y_th, mask_minority)
        m_w = compute_group_metrics(y_test, y_th, mask_white)
        di_race = m_min["selection_rate"] / m_w["selection_rate"] if m_w["selection_rate"] > 0 else 1.0
        
        threshold_results.append({
            "Threshold": round(th, 2),
            "Global_Recall": round(global_rec * 100, 1),
            "Global_Precision": round(global_prec * 100, 1),
            "Gender_Disparate_Impact": round(di_g, 3),
            "Race_Disparate_Impact": round(di_race, 3)
        })

    df_thresholds = pd.DataFrame(threshold_results)
    csv_thresh_path = os.path.join(OUTPUT_DIR, "threshold_optimization_audit.csv")
    df_thresholds.to_csv(csv_thresh_path, index=False)
    print(f"  ✓ Saved Threshold Audit: {csv_thresh_path}")

    # 6. Visual Multi-Panel Audit Plot
    print("\n[6/6] Generating Publication-Quality Multi-Panel Fairness Visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Plot 1: Disparate Impact
    dims = df_fairness["Dimension"].tolist()
    di_vals = df_fairness["Disparate_Impact_Ratio"].tolist()
    colors = ["#10b981" if 0.80 <= v <= 1.25 else "#f59e0b" for v in di_vals]
    b1 = axes[0, 0].bar(dims, di_vals, color=colors, width=0.45, edgecolor="#1f2937", linewidth=0.8)
    axes[0, 0].axhline(0.80, color="#ef4444", linestyle="--", linewidth=1.5, label="80% Lower Bound (0.80)")
    axes[0, 0].axhline(1.25, color="#ef4444", linestyle="--", linewidth=1.5, label="125% Upper Bound (1.25)")
    axes[0, 0].axhline(1.00, color="#6b7280", linestyle=":", linewidth=1.0, label="Parity Target (1.00)")
    axes[0, 0].set_ylim(0, 1.7)
    axes[0, 0].set_ylabel("Disparate Impact Ratio", fontsize=11)
    axes[0, 0].set_title("A. Disparate Impact Ratio vs. Four-Fifths Standard", fontsize=12, fontweight="bold")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    axes[0, 0].grid(axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(b1, di_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.3f}", ha="center", fontweight="bold", fontsize=9.5)

    # Plot 2: Equal Opportunity Parity
    x = np.arange(len(dims))
    w = 0.28
    priv_rec = df_fairness["Priv_Recall_TPR"].tolist()
    unpriv_rec = df_fairness["Unpriv_Recall_TPR"].tolist()
    axes[0, 1].bar(x - w/2, priv_rec, w, label="Privileged Group", color="#3b82f6", edgecolor="#1f2937", linewidth=0.8)
    axes[0, 1].bar(x + w/2, unpriv_rec, w, label="Protected Slice", color="#ec4899", edgecolor="#1f2937", linewidth=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(dims, fontsize=9.5, rotation=10)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_ylabel("Recall / True Positive Rate (%)", fontsize=11)
    axes[0, 1].set_title("B. Equal Opportunity: Recall Parity per Protected Slice", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="lower right", fontsize=9)
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.6)
    for i in range(len(dims)):
        axes[0, 1].text(x[i] - w/2, priv_rec[i] + 1.5, f"{priv_rec[i]}%", ha="center", fontsize=8.5, fontweight="bold")
        axes[0, 1].text(x[i] + w/2, unpriv_rec[i] + 1.5, f"{unpriv_rec[i]}%", ha="center", fontsize=8.5, fontweight="bold")

    # Plot 3: Statistical Parity Difference
    spd_vals = df_fairness["Statistical_Parity_Diff"].tolist()
    axes[1, 0].barh(dims, spd_vals, color=["#10b981" if abs(v) <= 0.10 else "#f59e0b" for v in spd_vals], edgecolor="#1f2937", linewidth=0.8)
    axes[1, 0].axvline(0.00, color="black", linestyle="-", linewidth=1.0)
    axes[1, 0].axvline(-0.10, color="#ef4444", linestyle="--", linewidth=1.2, label="Acceptable Tolerance (±0.10)")
    axes[1, 0].axvline(0.10, color="#ef4444", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xlabel("Statistical Parity Difference (Unpriv - Priv)", fontsize=11)
    axes[1, 0].set_title("C. Statistical Parity Difference (Selection Rate Gap)", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="lower right", fontsize=9)
    axes[1, 0].grid(axis="x", linestyle=":", alpha=0.6)
    for i, v in enumerate(spd_vals):
        axes[1, 0].text(v + (0.005 if v >= 0 else -0.025), i, f"{v:+.3f}", va="center", fontweight="bold", fontsize=9)

    # Plot 4: Threshold Recalibration
    th_x = df_thresholds["Threshold"]
    axes[1, 1].plot(th_x, df_thresholds["Global_Recall"], marker="o", color="#2563eb", linewidth=2.0, label="Global Recall (%)")
    axes[1, 1].plot(th_x, df_thresholds["Global_Precision"], marker="s", color="#10b981", linewidth=2.0, label="Global Precision (%)")
    axes[1, 1].axvline(0.50, color="#ef4444", linestyle="--", linewidth=1.5, label="Default Threshold (0.50)")
    axes[1, 1].set_xlabel("Decision Threshold", fontsize=11)
    axes[1, 1].set_ylabel("Performance (%)", fontsize=11)
    axes[1, 1].set_title("D. Accuracy vs. Threshold Recalibration", fontsize=12, fontweight="bold")
    axes[1, 1].legend(loc="center right", fontsize=9)
    axes[1, 1].grid(linestyle=":", alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "01_comprehensive_fairness_audit.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved Visual Audit: {chart_path}")

    print("\n" + "=" * 80)
    print("✨ EXPANDED FAIRNESS AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_phase4_audit()
