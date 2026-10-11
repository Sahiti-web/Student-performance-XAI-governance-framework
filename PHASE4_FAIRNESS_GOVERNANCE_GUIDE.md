# Phase 4: Algorithmic Fairness Evaluation & Ethical Governance Framework

## Overview
This branch (`phase-4/fairness-governance`) establishes **Phase 4: Algorithmic Fairness Auditing and Ethical AI Governance Policy** for student dropout prediction in technical secondary education.

Predictive machine learning models in high-stakes public education risk perpetuating or exacerbating historical inequities. This framework establishes an institutional auditing methodology to guarantee that the champion models operate fairly across protected demographic groups and comply with international educational AI ethics standards.

---

## 1. Protected Sensitive Demographic Attributes
Audits are conducted across the following sensitive attributes:
1. **Gender (`genero`)**:
   - Privileged / Baseline: Male (`genero_M`)
   - Unprivileged / Protected: Female (`genero_F`)
2. **Race / Ethnicity (`raca`)**:
   - Privileged / Baseline: White (`Branca`)
   - Unprivileged / Protected: Black and Mixed-race (`Preta`, `Parda`, `Indígena`)
3. **Socioeconomic Status (`rendabruta`)**:
   - Privileged / Baseline: Above median family income
   - Unprivileged / Protected: Below median / lowest income quintile

---

## 2. Core Quantitative Fairness Metrics

### A. Disparate Impact (Four-Fifths / 80% Rule)
$$\text{Disparate Impact} = \frac{P(\hat{Y}=1 \mid A = \text{unprivileged})}{P(\hat{Y}=1 \mid A = \text{privileged})}$$
* **Threshold**: Must fall within $[0.80, 1.25]$ to confirm no illegal disparate adverse impact.

### B. Statistical Parity Difference (Demographic Parity)
$$\text{SPD} = P(\hat{Y}=1 \mid A = \text{unprivileged}) - P(\hat{Y}=1 \mid A = \text{privileged})$$
* **Target**: $\approx 0.00$ ($\pm 0.10$). Checks whether dropout prediction frequency is equitable.

### C. Equal Opportunity Difference (True Positive Rate Parity)
$$\text{EOD} = \text{Recall}_{\text{unprivileged}} - \text{Recall}_{\text{privileged}}$$
* **Target**: $\approx 0.00$. Crucial for student safety: guarantees that dropouts in marginalized groups are caught with the same high sensitivity (Recall ≥ 84%) as the majority group.

### D. Average Odds Difference (Equalized Odds)
$$\text{AOD} = \frac{1}{2} \Big( (\text{FPR}_{\text{unprivileged}} - \text{FPR}_{\text{privileged}}) + (\text{TPR}_{\text{unprivileged}} - \text{TPR}_{\text{privileged}}) \Big)$$
* Evaluates whether both false alarms and detection accuracy are balanced across demographic strata.

---

## 3. Institutional Governance Policy & Human-in-the-Loop Protocol

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AI GOVERNANCE LIFECYCLE                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. EARLY WARNING ONLY (Advisory System)                                     │
│    • Model outputs are strictly advisory risk indicators for counselors.    │
│    • Automated disciplinary or enrollment actions are strictly PROHIBITED.  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. EXPLAINABILITY-DRIVEN INTERVENTION                                       │
│    • Counselors must review SHAP/LIME factor attributions prior to student  │
│      outreach to ensure intervention addresses root cause (academic vs aid).│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. CONTINUOUS AUDIT & DRIFT RECALIBRATION                                   │
│    • Bi-annual re-auditing of Disparate Impact and Equal Opportunity.       │
│    • Continuous monitoring of demographic shifts and concept drift.         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Execution Pipeline
To run the automated algorithmic fairness audit:
```bash
python phase4_fairness_audit_pipeline.py
```
*Outputs generated in `phase4_outputs/`:*
* `fairness_metrics_summary.csv`: Tabular audit of Disparate Impact, Demographic Parity, and Equal Opportunity across all protected groups.
* `01_fairness_disparate_impact_chart.png`: Visual bar chart comparing demographic groups against the 80% fairness threshold.
* `02_equal_opportunity_recall_parity.png`: Sensitivity parity across demographic slices.
