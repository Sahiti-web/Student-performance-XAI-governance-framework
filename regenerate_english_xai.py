"""
Regenerate Phase 3 SHAP visual plots and attribution tables with English feature labels.
"""
import os
import sys
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BUNDLE_PATH = "phase3_bundle.pkl"
OUTPUT_DIR = "phase3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# English Mapping Dictionary
FEATURE_MAP_ENGLISH = {
    "ira": "Academic Performance (GPA / IRA)",
    "reprovacoes": "Course Failures (Count)",
    "anoingresso": "Enrollment Year",
    "percentual_frequencia": "Attendance Rate (%)",
    "modalidade": "Course Modality",
    "idade": "Student Age",
    "rendabruta": "Family Gross Income",
    "trabalha": "Employment Status",
    "mae_nivel_escolaridade": "Mother's Education Level",
    "companhia_domiciliar": "Household Companionship",
    "pai_nivel_escolaridade": "Father's Education Level",
    "quantidade_notebooks": "Laptops at Home",
    "genero_M": "Gender: Male",
    "genero_F": "Gender: Female",
    "tipo_area_residencial": "Residential Area (Urban/Rural)"
}

def main():
    print("Loading bundle...")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    fn_pt = bundle["feature_names"]
    X_test = bundle["X_test"]

    # Translate feature names for display
    fn_en = [FEATURE_MAP_ENGLISH.get(f, f) for f in fn_pt]
    df_test_en = pd.DataFrame(X_test, columns=fn_en)

    print("Computing TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(model)
    n_sample = min(800, len(X_test))
    shap_raw = explainer.shap_values(X_test[:n_sample])

    if isinstance(shap_raw, list):
        shap_dropout = shap_raw[1]
    elif len(shap_raw.shape) == 3:
        shap_dropout = shap_raw[:, :, 1]
    else:
        shap_dropout = shap_raw

    # 1. Global Beeswarm Plot in English
    print("Saving 01_shap_global_beeswarm.png...")
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_dropout, df_test_en.iloc[:n_sample], show=False, plot_type="dot")
    plt.title("Global SHAP Beeswarm: Feature Impact on Student Dropout Risk", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_shap_global_beeswarm.png"), dpi=300)
    plt.close()

    # 2. Global Feature Importance Bar Plot in English
    print("Saving 02_shap_feature_importance.png...")
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_dropout, df_test_en.iloc[:n_sample], show=False, plot_type="bar")
    plt.title("Global Feature Importance: Mean |SHAP Value| (Dropout Class)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_shap_feature_importance.png"), dpi=300)
    plt.close()

    # 3. Dependence Plot in English
    try:
        ira_idx = fn_en.index("Academic Performance (GPA / IRA)")
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(ira_idx, shap_dropout, df_test_en.iloc[:n_sample].values, feature_names=fn_en, show=False)
        plt.title("SHAP Dependence: Academic Performance (GPA) vs. Attendance Interaction", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "03_shap_dependence_ira.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Dependence plot notice:", e)

    # 4. Save Translated CSV Attributions Table
    mean_abs = np.abs(shap_dropout).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    df_rank_en = pd.DataFrame({
        "Rank": np.arange(1, len(fn_en) + 1),
        "Feature_Original": [fn_pt[i] for i in order],
        "Feature": [fn_en[i] for i in order],
        "Mean_Absolute_SHAP": [mean_abs[i] for i in order]
    })
    csv_out = os.path.join(OUTPUT_DIR, "shap_feature_attributions.csv")
    df_rank_en.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    # 5. Regenerate interactive students JSON with English features
    print("Updating interactive_students_xai.json with English feature names...")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)
    y_test = bundle["y_test"]

    cases = [0, 43, 85, 37, 102, 185, 312, 1, 25, 60, 14, 52]
    students_data = []

    for idx in cases:
        raw_s = explainer.shap_values(X_test[idx:idx+1])
        if isinstance(raw_s, list):
            s_shap = raw_s[1][0]
        elif len(raw_s.shape) == 3:
            s_shap = raw_s[0, :, 1]
        else:
            s_shap = raw_s[0]

        feat_vals = X_test[idx]
        sorted_idx = np.argsort(np.abs(s_shap))[::-1][:6]
        drivers = []
        for si in sorted_idx:
            drivers.append({
                "feature": fn_en[si],
                "feature_orig": fn_pt[si],
                "value": round(float(feat_vals[si]), 2),
                "shap": round(float(s_shap[si]), 4),
                "direction": "increases_risk" if s_shap[si] > 0 else "protective"
            })

        # Prescriptions
        interventions = []
        f_dict = dict(zip(fn_pt, feat_vals))
        if f_dict.get("ira", 10) < 5.0:
            interventions.append({"urgency": "High", "title": "Academic Remediation", "text": "Mandatory peer-tutoring and foundations workshop."})
        if f_dict.get("reprovacoes", 0) >= 1.0:
            interventions.append({"urgency": "Critical", "title": "Course Retake Plan", "text": "Academic advising consultation; adjust semester study load."})
        if f_dict.get("percentual_frequencia", 100) < 75.0:
            interventions.append({"urgency": "Immediate", "title": "Attendance Outreach", "text": "Social welfare officer outreach to investigate transportation barriers."})
        if f_dict.get("rendabruta", 5000) < 1000.0:
            interventions.append({"urgency": "Moderate", "title": "Financial Aid Priority", "text": "Expedited eligibility check for transport grants and meal stipends."})
        if f_dict.get("quantidade_notebooks", 1) == 0:
            interventions.append({"urgency": "High", "title": "Digital Inclusion Loan", "text": "Issue institutional loaner laptop and campus lab card."})
        if not interventions:
            interventions.append({"urgency": "Standard", "title": "Normal Progress Tracking", "text": "Maintain bi-monthly check-ins; potential peer-tutor candidate."})

        students_data.append({
            "id": int(idx),
            "prob": round(float(probs[idx]) * 100, 1),
            "pred": "Dropout" if preds[idx] == 1 else "Completed",
            "actual": "Dropout" if y_test[idx] == 1 else "Completed",
            "features_en": {fn_en[k]: round(float(feat_vals[k]), 2) for k in range(len(fn_en))},
            "features_raw": {fn_pt[k]: round(float(feat_vals[k]), 2) for k in range(len(fn_pt))},
            "drivers": drivers,
            "interventions": interventions
        })

    json_out = "phase4_outputs/interactive_students_xai.json"
    with open(json_out, "w", encoding="utf-8") as f:
        import json
        json.dump(students_data, f, indent=2)
    print(f"Saved {json_out}")
    print("Done!")

if __name__ == "__main__":
    main()
