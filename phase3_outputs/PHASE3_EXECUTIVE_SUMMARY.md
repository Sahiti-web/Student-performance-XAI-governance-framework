# Phase 3 Executive Summary: Explainable AI & Institutional Governance

## 1. Executive Summary & Objective
Phase 3 translates the champion machine learning model from Phase 2 (`RandomForestClassifier`, 84.8% recall, 92.9% accuracy) into a transparent, actionable decision-support tool for high school educators and counselors. 

By applying **SHAP (Shapley Additive Explanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**, this phase resolves the "black box" concern, identifying both global institutional bottlenecks and personalized student risk factors.

---

## 2. Top Global Dropout Indicators (SHAP Analysis)
Across held-out student cohorts, the top 5 empirical drivers of dropout risk are:

1. **`ira`**: Mean |SHAP| = 0.2717
2. **`reprovacoes`**: Mean |SHAP| = 0.0596
3. **`anoingresso`**: Mean |SHAP| = 0.0415
4. **`percentual_frequencia`**: Mean |SHAP| = 0.0305
5. **`modalidade`**: Mean |SHAP| = 0.0263

### Key Findings:
1. **Academic Performance Index (`ira`) is the primary driver**: Students with an IRA below 5.0 experience an exponential increase in predicted dropout probability.
2. **Subject Failures (`reprovacoes`) act as a secondary catalyst**: Failing one or more courses strongly amplifies risk, serving as an early-warning checkpoint before attendance drop-off.
3. **Attendance (`percentual_frequencia`) exhibits a strict threshold behavior**: Falling below 75% attendance triggers significant risk increases.

---

## 3. Archetypal Student Case Studies & Counselor Guidance

### Student #0 — High-Risk Student
* **Assessed Risk**: `99.4%` probability of dropping out (Predicted: `Dropped Out`, Actual: `Dropped Out`)
* **Key Risk Aggravators**: `ira` (+0.2831), `reprovacoes` (+0.1134), `percentual_frequencia` (+0.0562), `modalidade` (+0.0235)
* **Prescribed Interventions**:
  - **High Urgency**: Assign mandatory peer-tutoring, remedial workshops in foundation courses, and bi-weekly mentor check-ins.
  - **Critical Urgency**: Urgent academic advising consultation; adjust semester course load and map out recovery study plan.
  - **Moderate Urgency**: Expedited screening for institutional student financial aid, food subsidies, and transportation grants.
  - **High Urgency**: Issue institutional loaner laptop and ensure access to campus computer labs and digital library resources.
  - **Low Urgency**: Connect with adult-learner peer support networks and evening study groups.

### Student #37 — Borderline / At-Risk Student
* **Assessed Risk**: `50.1%` probability of dropping out (Predicted: `Dropped Out`, Actual: `Completed`)
* **Key Risk Aggravators**: `percentual_frequencia` (+0.0746), `modalidade` (+0.0235)
* **Prescribed Interventions**:
  - **Critical Urgency**: Urgent academic advising consultation; adjust semester course load and map out recovery study plan.
  - **Low Urgency**: Connect with adult-learner peer support networks and evening study groups.

### Student #1 — Low-Risk / Resilient Student
* **Assessed Risk**: `3.4%` probability of dropping out (Predicted: `Completed`, Actual: `Completed`)
* **Key Risk Aggravators**: `pai_nivel_escolaridade` (+0.0070)
* **Prescribed Interventions**:
  - **Moderate Urgency**: Expedited screening for institutional student financial aid, food subsidies, and transportation grants.


---

## 4. Phase 3 Deliverables
* Interactive Dashboard: `phase3_outputs/phase3_xai_dashboard.html`
* Global Visualizations: `phase3_outputs/01_shap_global_beeswarm.png`, `phase3_outputs/02_shap_feature_importance.png`, `phase3_outputs/03_shap_dependence_ira.png`
* Local Archetype Plots: `phase3_outputs/04_shap_*.png`
* Attribution Table: `phase3_outputs/shap_feature_attributions.csv`
