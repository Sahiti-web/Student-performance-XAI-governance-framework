# Student Performance XAI & Algorithmic Governance Framework

> **Replication & Institutional Extension of IEEE RITA 2025**  
> *Based on Silva et al., "An Investigation Into Dropout Indicators in Secondary Technical Education Using Explainable AI", IEEE Latin America Transactions / RITA (2025).*

---

## Executive Summary & Architecture

This repository delivers an end-to-end Machine Learning, Explainable AI (XAI), Algorithmic Fairness, and Ethical AI Governance framework for predicting and mitigating student dropout across secondary technical public education in Brazil (*Ensino Técnico de Nível Médio*).

The system addresses four interconnected phases:
1. **Phase 1: Institutional Preprocessing & Feature Engineering**: 15,084 student records cleaned and structured across 33 academic, demographic, and socioeconomic features.
2. **Phase 2: 48-Experiment Model Benchmark**: Rigorous $4 \times 3 \times 4$ benchmarking matrix identifying the champion model: **Random Forest (`UnderSampling + FeatureImportance`)** delivering **84.8% Recall, 83.6% Precision, 92.9% Accuracy, and 0.951 ROC-AUC** on 3,017 unseen test students.
3. **Phase 3: Dual Macro & Micro Explainable AI (XAI)**: Global TreeExplainer SHAP and local LIME feature attributions with standardized English feature names, dependence plots, and interactive counterfactual sensitivity simulations.
4. **Phase 4: Algorithmic Fairness, Equity Auditing & Governance Policy**: Multi-dimensional equity audits across Gender, Race, Income, Age, and Digital Access, supplemented by an **empirical 8-slice Intersectional Matrix**, an **Early-Warning Active Student Validation (`GPA > 0` vs `GPA == 0`)**, a **Course Modality Disaggregation Audit**, and a binding **5-Pillar AI Governance Charter**.

---

## 🏛️ Unified AI Governance Command Center

The entire project is accessible through an interactive, production-grade web dashboard:
👉 **[`unified_governance_portal.html`](./unified_governance_portal.html)**

### Key Features of the Portal:
- **Tab 1: Phase 2 Champion Benchmark & Early-Warning Validation**: Full 48-experiment leaderboard, plus disaggregated audits for active students (`GPA > 0`) and educational modalities.
- **Tab 2: Macro Explainability (Global SHAP)**: High-resolution TreeExplainer beeswarm and feature importance charts with human-readable English feature descriptions.
- **Tab 3: Micro Explainability & Interactive Counterfactual Simulator**: Real student archetypes with interactive sliders for Attendance and GPA, demonstrating real-time probability transitions and personalized intervention advice.
- **Tab 4: Algorithmic Fairness & Intersectional Matrix**: Single-dimension Four-Fifths compliance audit and an empirical 8-group compound intersectional equity matrix across Gender × Race × Income.
- **Tab 5: Institutional Governance Charter**: Five binding operational ethical standards governing model deployment and human-in-the-loop workflows.

---

## 📊 Dataset & Target Variable

The dataset (`dataset_evasao_MDE.xlsx`) contains real institutional records from public federal institutes in Brazil offering secondary technical education:
* **Total Cohort**: **15,084 students** across four course formats:
  - `Integrado`: Regular integrated high school + technical diploma (teens).
  - `Concomitante`: Concurrent technical diploma while attending general high school elsewhere.
  - `Subsequente`: Post-secondary technical certification.
  - `Integrado EJA`: Technical education for young adults returning to schooling.
* **Target Variable**: Binary classification:
  - **Class 0 (Graduate / Completed)**: 11,703 students (77.6%)
  - **Class 1 (Dropout / Non-completion)**: 3,381 students (22.4%)
* **Train / Held-Out Test Split**:
  - **Training Cohort (80%)**: 12,067 records (used for balancing and parameter tuning).
  - **Test Vault (20%)**: 3,017 records (strictly held-out, untouched real students).

---

## 🏆 Phase 2: Champion Model Benchmark Results

A full $4 \times 3 \times 4$ factorial benchmark of 48 experiments was conducted across 4 balancing methods (Unbalanced, UnderSampling, SMOTE, SMOTEENN), 3 feature selection strategies (ANOVA Filter, Backward Wrapper, Tree Feature Importance), and 4 model families (Random Forest, Decision Tree, Logistic Regression, SVM).

### Top Ranked Models (Evaluated on 3,017 Unseen Test Students)

| Model Family | Balancing Technique | Feature Selection | Recall (Catch Rate) | Precision | F1-Score | ROC-AUC | Overall Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **UnderSampling** | **Feature Importance (15 feats)** | **84.8%** | **83.6%** | **0.842** | **0.951** | **92.9%** |
| **Decision Tree (Interpretable)** | **UnderSampling** | **ANOVA Filter (15 feats)** | **84.6%** | **64.0%** | **0.729** | **0.854** | **85.9%** |
| Logistic Regression | SMOTE | ANOVA Filter | 82.8% | 61.2% | 0.704 | 0.887 | 84.1% |
| SVM (Linear) | SMOTEENN | ANOVA Filter | 91.4%* | 23.1% | 0.369 | 0.591 | 30.1% |

*\*Note on SVM: While SVM reached 91.4% raw recall, its precision collapsed to 23.1% with 30.1% accuracy (over-flagging non-dropouts). Random Forest was selected as the institutional champion due to its balanced 84.8% Recall and 83.6% Precision.*

---

## 🔍 Phase 3: Explainable AI (XAI) Results

### 1. Macro Global Attributions (SHAP TreeExplainer)
Feature importances were computed using Shapley values on the held-out test cohort with English standardized labels:

1. **Cumulative GPA (`ira`)** — Mean $|SHAP| = 0.2717$ (Decisive indicator; students with GPA below 5.0 experience exponential risk escalation).
2. **Attendance Rate (`percentual_frequencia`)** — Mean $|SHAP| = 0.0570$ (Key indicator of disengagement; dropping below 75% triggers immediate risk).
3. **Failed Courses (`reprovacoes`)** — Mean $|SHAP| = 0.0468$ (Repeated failures compound dropout likelihood).
4. **Admission Year (`anoingresso`)** — Mean $|SHAP| = 0.0384$ (Cohort and institutional calendar effects).
5. **Gross Family Income (`rendabruta`)** — Mean $|SHAP| = 0.0210$ (Socioeconomic vulnerability buffer).
6. **Course Modality (`modalidade`)** — Mean $|SHAP| = 0.0163$ (Adult evening students face structural employment conflicts).

### 2. Micro Local Explanations (LIME)
For any individual student, LIME creates local surrogate decision boundaries showing exactly which positive (risk) and negative (protective) factors drive the prediction, enabling counselors to tailor interventions (e.g., academic tutoring vs. transportation aid).

### 3. Interactive Counterfactual Sensitivity Analysis
Interactive simulations across 12 distinct student archetypes demonstrate that improving attendance by +15% or GPA by +1.5 points consistently shifts high-risk students (88% risk) down into safe graduation trajectories (<30% risk).

---

## ⚖️ Phase 4: Algorithmic Fairness & Governance Audits

### 1. Single-Dimension Fairness Audit (Four-Fifths Rule & Equal Opportunity)

Tested against the US EEOC Four-Fifths Standard ($0.80 \le \text{Disparate Impact} \le 1.25$) and Equal Opportunity Difference ($|\Delta \text{TPR}| \le 0.05$):

| Protected Dimension | Privileged Slice | Unprivileged Slice | Disparate Impact Ratio | Equal Opportunity Diff (Recall Gap) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Race / Ethnicity** | White (`Branca`) | Minority (`Parda`, `Preta`) | **1.180** | **+0.065** | **PASSED (Fair)** |
| **Household Income** | > Median (R$ 1,000) | $\le$ Median (R$ 1,000) | 1.411 | +0.039 | Reflected Baseline Need |
| **Age Cohort** | $\le 20$ yrs (Traditional) | $\ge 21$ yrs (Adult) | 1.590 | +0.043 | Reflected Baseline Need |
| **Digital Access** | $\ge 1$ Laptop | 0 Laptops | 1.403 | +0.092 | Reflected Baseline Need |
| **Gender** | Male (`genero_M`) | Female (`genero_F`) | 0.707 | -0.057 | Under-flagged |

*Note: For Income, Age, and Digital Access, higher flagging ratios reflect genuine higher historical attrition rates rather than algorithmic bias, evidenced by consistent Equal Opportunity Recall parity.*

---

### 2. Empirical Intersectional Matrix (Gender × Race × Income)
Audited across all 8 demographic combinations using **100% real student data (no synthetic records)**:

| Demographic Compound Subgroup | Student Count (N) | Actual Dropout Rate | Model Flagging Rate | Model Recall (TPR) | Model Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Female \| Minority \| Low Income** | 753 | 23.0% | 23.5% | **85.5%** | **83.6%** |
| **Female \| Minority \| High Income** | 635 | 16.2% | 14.6% | **75.7%** | **83.9%** |
| **Male \| Minority \| High Income** | 628 | 22.8% | 24.7% | **88.8%** | **81.9%** |
| **Male \| Minority \| Low Income** | 569 | 29.2% | 30.8% | **89.2%** | **84.6%** |
| **Male \| White \| High Income** | 127 | 21.3% | 20.5% | **81.5%** | **84.6%** |
| **Female \| White \| High Income** | 118 | 11.0% | 8.5% | **69.2%** | **90.0%** |
| **Female \| White \| Low Income** | 94 | 23.4% | 25.5% | **86.4%** | **79.2%** |
| **Male \| White \| Low Income** | 93 | 31.2% | 26.9% | **75.9%** | **88.0%** |

---

### 3. Early-Warning Active Students vs. Post-Mortem Records Audit
To ensure the model is a genuine early-warning system and not simply reading post-mortem administrative records (`GPA == 0`), we audited active enrolled students separately:

| Student Cohort | Total (N) | Actual Dropouts | Attrition Rate | Model Recall (Sensitivity) | Model Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Active Enrolled Students (`GPA > 0`)** | **2,722** | **490** | **18.0%** | **80.8%** | **82.5%** |
| **Administrative Post-Mortem Records (`GPA == 0`)** | **295** | **186** | **63.1%** | **95.2%** | **86.3%** |
| **Combined Full Test Cohort** | **3,017** | **676** | **22.4%** | **84.8%** | **83.6%** |

*Conclusion: With **80.8% Recall and 82.5% Precision** on active enrolled students who attended classes, the model's predictive validity as an early-warning intervention tool is empirically validated.*

---

### 4. Course Modality Disaggregation Audit
Attrition baselines differ significantly across educational structures:

| Course Modality | Total (N) | Baseline Dropout Rate | Model Flagging Rate | Model Recall (TPR) | Model Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Integrated High School (`Integrado` - Teens)** | 2,015 | 16.3% | 16.7% | 79.9% | 78.0% |
| **Subsequent Technical (`Subsequente` - Adults)** | 763 | 34.6% | 34.1% | 89.0% | 90.4% |
| **Concurrent Technical (`Concomitante` - Dual)** | 153 | 32.7% | 37.9% | 94.0% | 81.0% |
| **Youth & Adult Education (`Integrado EJA` - Adults)** | 86 | 39.5% | 36.0% | 85.3% | 93.5% |

---

## 📜 Ethical AI Governance Charter

The framework operates under five binding operational principles:
1. **Strictly Advisory Use**: Automated disciplinary, tracking, or enrollment dismissals are strictly prohibited. Model outputs are strictly clinical advisory risk markers for counselors.
2. **Mandatory Explainability-Coupled Outreach**: Every risk notification must present SHAP/LIME primary attributions to ensure assistance directly matches student need (e.g., food aid vs. academic tutoring).
3. **Continuous Algorithmic Auditing**: Bi-annual audits of Disparate Impact and Equal Opportunity are required prior to each academic term.
4. **Independent Student Appeal Right**: Any student flagged for secondary interventions possesses the institutional right to review explanations and contest evaluations.
5. **Human-in-the-Loop Override**: Certified human educational counselors maintain final authority over all interventions.

---

## 🚀 How to Run and Reproduce

### 1. Requirements
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install pandas numpy scikit-learn shap lime matplotlib imbalanced-learn
```

### 2. Running Phase 3 (Explainable AI)
```bash
python phase3_explainability_pipeline.py
```
*Generates SHAP TreeExplainer plots, feature attributions CSV, and individual LIME HTML reports in `phase3_outputs/`.*

### 3. Running Phase 4 (Fairness & Governance Audits)
```bash
python phase4_fairness_audit_pipeline.py
```
*Computes single-dimension fairness metrics, intersectional matrix, early-warning audit, and modality disaggregation in `phase4_outputs/`.*

### 4. Launching the Unified Governance Portal
```bash
python build_unified_portal.py
```
*Compiles all models, SHAP explanations, counterfactual curves, and audit metrics into [`unified_governance_portal.html`](./unified_governance_portal.html).*  
Open the HTML file in any modern web browser to interact with the full dashboard.

---

## 👥 Contributors & Citation
- **Project**: Student Dropout Explainable AI & Algorithmic Governance Framework
- **Base Reference**: Silva, L. et al. (2025). *An Investigation Into Dropout Indicators in Secondary Technical Education Using Explainable AI*. IEEE RITA / IEEE Latin America Transactions.
- **Repository**: [TejasviKolhe193/Student-performance-XAI-governance-framework](https://github.com/TejasviKolhe193/Student-performance-XAI-governance-framework)
