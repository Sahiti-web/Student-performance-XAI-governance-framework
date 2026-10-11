"""
Phase 3: Explainable AI (XAI) Pipeline — SHAP & LIME Interpretability
======================================================================
Replicates the explainability methodology of:
  Silva et al., "An Investigation Into Dropout Indicators in 
  Secondary Technical Education Using Explainable AI" (IEEE RITA, 2025)

Capabilities:
  1. Load champion Random Forest model & test data from `phase3_bundle.pkl`
  2. Compute global SHAP attributions (TreeExplainer)
  3. Generate global feature importance & summary plots
  4. Compute local student-level explanations via SHAP waterfall & LIME
  5. Produce counselor-actionable interpretability reports
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

import shap
from lime.lime_tabular import LimeTabularExplainer

# Console encoding for Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

BUNDLE_PATH = "phase3_bundle.pkl"
OUTPUT_DIR = "phase3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_phase3_xai():
    print("=" * 72)
    print("PHASE 3: EXPLAINABLE AI (XAI) PIPELINE")
    print("=" * 72)

    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Missing bundle: {BUNDLE_PATH}. Please run Phase 2 training first.")

    print(f"\n[1/5] Loading champion model bundle from '{BUNDLE_PATH}'...")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    print(f"  Model Architecture: {type(model).__name__}")
    print(f"  Features ({len(feature_names)}): {', '.join(feature_names)}")
    print(f"  Test Cohort: {X_test.shape[0]} students")

    # ── 2. GLOBAL SHAP EXPLANATIONS ──────────────────────────────────────
    print("\n[2/5] Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for class 1 (Dropout)
    sample_size = min(500, len(X_test))
    print(f"  Computing SHAP values on {sample_size} representative student samples...")
    shap_values = explainer.shap_values(X_test[:sample_size])

    # TreeExplainer for binary RF returns a list of 2 arrays [class_0, class_1] or a 3D array
    if isinstance(shap_values, list):
        shap_values_dropout = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_dropout = shap_values[:, :, 1]
    else:
        shap_values_dropout = shap_values

    # Save SHAP Summary Beeswarm Plot
    print("\n[3/5] Generating SHAP Global Visualizations...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_dropout,
        X_test[:sample_size],
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    summary_path = os.path.join(OUTPUT_DIR, "01_shap_global_beeswarm.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {summary_path}")

    # Save SHAP Global Bar Importance Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_dropout,
        X_test[:sample_size],
        feature_names=feature_names,
        show=False,
        plot_type="bar"
    )
    bar_path = os.path.join(OUTPUT_DIR, "02_shap_feature_importance.png")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {bar_path}")

    # ── 3. LIME LOCAL EXPLANATIONS ───────────────────────────────────────
    print("\n[4/5] Initializing LIME Tabular Explainer...")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test,
        feature_names=feature_names,
        class_names=["Completed", "Dropped Out"],
        mode="classification",
        random_state=42
    )

    # Find highest-risk dropout student in sample
    probabilities = model.predict_proba(X_test[:sample_size])
    high_risk_idx = int(np.argmax(probabilities[:, 1]))
    high_risk_prob = probabilities[high_risk_idx, 1]
    actual_status = "Dropped Out" if y_test[high_risk_idx] == 1 else "Completed"

    print(f"  Flagged High-Risk Student #{high_risk_idx}:")
    print(f"    Dropout Probability: {high_risk_prob * 100:.1f}% | Actual Outcome: {actual_status}")

    lime_exp = lime_explainer.explain_instance(
        data_row=X_test[high_risk_idx],
        predict_fn=model.predict_proba,
        num_features=8
    )

    lime_html_path = os.path.join(OUTPUT_DIR, f"03_lime_student_{high_risk_idx}_report.html")
    lime_exp.save_to_file(lime_html_path)
    print(f"  ✓ Saved LIME Local Report: {lime_html_path}")

    print("\n[5/5] Generating Counselor Actionable Insight Table...")
    mean_abs_shap = np.abs(shap_values_dropout).mean(axis=0)
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    csv_importance_path = os.path.join(OUTPUT_DIR, "shap_feature_attributions.csv")
    df_importance.to_csv(csv_importance_path, index=False)
    print(f"  ✓ Saved Attributions Table: {csv_importance_path}")

    print("\nTop 5 Drivers of Dropout Risk Identified by SHAP:")
    for idx, row in df_importance.head(5).iterrows():
        print(f"  #{idx+1}: {row['Feature']:<25s} | Mean |SHAP| = {row['Mean_Absolute_SHAP']:.4f}")

    print("\n" + "=" * 72)
    print("✅ PHASE 3 EXPLAINABLE AI PIPELINE COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    run_phase3_xai()
