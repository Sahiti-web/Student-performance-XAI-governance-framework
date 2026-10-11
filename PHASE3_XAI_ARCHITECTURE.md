# Phase 3: Explainable AI (XAI) Architecture & Implementation Guide

## Overview
This branch (`phase-3/explainable-ai`) implements **Phase 3: Explainable Artificial Intelligence (XAI)** of the Student Dropout Prediction & Governance Framework.

Building directly upon the champion models identified in Phase 2, this phase addresses the **"Black Box" Problem** in educational predictive modeling by answering two central operational questions:
1. **Global Interpretability**: What system-wide factors most strongly drive student dropout risk across the entire population?
2. **Local Interpretability**: For a specific individual student flagged at high risk of dropping out, *why* was that prediction made, and what targeted academic or socioeconomic interventions can counselors apply?

This methodology directly replicates and extends the explainability framework established in:
> **Silva et al., "An Investigation Into Dropout Indicators in Secondary Technical Education Using Explainable AI" (IEEE RITA, 2025).**

---

## 1. Input Bundle (`phase3_bundle.pkl`)
Phase 3 directly consumes the serialized output bundle generated at the conclusion of Phase 2:
* **Champion Model**: Tuned `RandomForestClassifier` (Recall: 0.848, AUC: 0.951, Accuracy: 92.9%).
* **Features Selected ($k=15$)**:
  - Academic: `ira` (GPA), `reprovacoes` (Course failures), `percentual_frequencia` (Attendance rate).
  - Background & Cohort: `anoingresso` (Enrollment year), `modalidade` (Course modality), `idade` (Age).
  - Socioeconomic: `rendabruta` (Income), `mae_nivel_escolaridade`, `pai_nivel_escolaridade`, `quantidade_notebooks`, `tipo_area_residencial`, `trabalha`, `companhia_domiciliar`.
  - Protected Attributes: `genero_F`, `genero_M`.
* **Evaluation Samples**: 3,017 held-out test students (`X_test`, `y_test`).

---

## 2. Core Explainability Techniques

### A. SHAP (SHapley Additive exPlanations)
Based on cooperative game theory, SHAP allocates fair credit for model predictions:
* **Explainer Algorithm**: `shap.TreeExplainer` optimized for tree-based ensemble models.
* **Global Summary Plot**: Visualizes the directional impact and magnitude of all 15 features across all 3,017 students.
* **Local Waterfall & Force Plots**: Deconstructs an individual student's risk probability into positive and negative push factors relative to the baseline base value.

### B. LIME (Local Interpretable Model-agnostic Explanations)
* **Explainer Algorithm**: `lime.lime_tabular.LimeTabularExplainer`.
* Generates locally linear surrogate models around individual student profiles to produce human-readable decision rules (e.g., *"ira <= 4.2 increases dropout probability by +38%"*).

### C. White-Box Surrogate Model (Decision Tree)
* Complements the Random Forest predictions with clear, deterministic `if-then` decision rules from the champion Decision Tree surrogate model (Recall: 0.846).

---

## 3. Counselor Intervention Translation Matrix

| Detected Risk Factor | Model Attribution Direction | Recommended Institutional Intervention |
| :--- | :--- | :--- |
| **Low Academic Performance (`ira` < 5.0)** | Strong positive push for Dropout | Mandatory peer-tutoring and remedial classes in core subjects. |
| **High Subject Failures (`reprovacoes` ≥ 2)** | Strong positive push for Dropout | Academic advising check-in, course retake planning. |
| **Low Attendance (`percentual_frequencia` < 75%)** | Strong positive push for Dropout | Social worker outreach to investigate commuting or family hardship. |
| **Low Household Income (`rendabruta` low)** | Moderate positive push for Dropout | Priority referral for student financial aid and transport stipends. |
| **Working While Studying (`trabalha` = True)** | Moderate positive push for Dropout | Flexible scheduling and evening tutoring support. |

---

## 4. Execution Pipeline
To run the Phase 3 XAI evaluation script:
```bash
python phase3_explainability_pipeline.py
```
*Outputs generated in `phase3_outputs/`:*
* `shap_global_summary.png`: Beeswarm summary plot of all features.
* `shap_feature_importance.png`: Mean absolute SHAP value bar chart.
* `lime_student_case_*.png` / `.html`: Individual student local explanation reports.
