# Changelog

## Phase 1: Data Preparation & Translation (Completed)
- **Data Loading & Filtering**: Extracted subset matching secondary technical education (15,084 students).
- **Target Construction**: Created binary target variable corresponding to Completed (Class 0: 77.6%) vs. Dropped Out (Class 1: 22.4%), matching base paper exactly.
- **Missing Value Handling**: Dropped columns with >20% missing values (`dataconclusao`, `forma_acesso_seletivo`, `razao_ausencia_educacional`, `pontuacao_seletivo`). Filled minor missing values in `quantidade_computadores` and `quantidade_notebooks` with median.
- **Categorical Normalization**: Normalized case variants in `raca`.
- **IQR Outlier Winsorization**: Applied 1.5×IQR winsorization (cap/clip) specifically to `idade`, `rendabruta`, `quantidade_computadores`, `quantidade_notebooks`, and `qtd_filhos` to handle extreme values without dropping rows (preserving 15,084 records and the 78/22 class balance). Explicitly excluded causal variables like `reprovacoes`, `ira`, `percentual_frequencia`.
- **Encoding**: 
  - One-hot encoded `genero`, `raca`, `estado_civil`.
  - Ordinal encoded `mae_nivel_escolaridade`, `pai_nivel_escolaridade`.
  - Label encoded remaining categoricals.
- **Train/Test Split**: Stratified 80/20 split based on the target class.
- **Data Translation**: Created a complete Portuguese-to-English translation layer (column names and categorical values) and generated English-translated datasets, saving mapping artifacts for downstream XAI labels. Addressed typo in dataset for 'Pós-gradução completa'.

## Phase 2: Model Training, Balancing & Evaluation (Completed)
- **Status**: Completed full 48-experiment benchmark grid (4 balancing strategies × 3 feature selection methods × 4 ML models).
- **Champion Models**:
  - Primary Predictor: **Random Forest** (UnderSampling + FeatureImportance) — **Recall: 0.848**, Precision: 0.836, F1: 0.842, AUC-ROC: **0.951**, Accuracy: **92.9%**.
  - White-Box Surrogate: **Decision Tree** (UnderSampling + ANOVA) — **Recall: 0.846**, Precision: 0.640, F1: 0.729, AUC-ROC: **0.854**, Accuracy: **85.9%**.
- **Deliverables**: Generated `experiment_results.csv`, `best_model_per_algorithm.csv`, `top_10_models.csv`, visual analytics (`01_model_comparison.png` through `05_top_features_importance.png`), interactive HTML dashboard (`phase2_dashboard.html`), and Phase 3 export bundle (`phase3_bundle.pkl`).

## Phase 3: Explainable AI (XAI) (Scaffolded & Ready)
- **Branch**: `phase-3/explainable-ai`
- **Scope**: Local & global interpretability via SHAP (TreeExplainer, beeswarm, waterfall) and LIME on the champion Random Forest model using `phase3_bundle.pkl`.
- **Implementation**: Provided `PHASE3_XAI_ARCHITECTURE.md` and `phase3_explainability_pipeline.py`.

## Phase 4: Algorithmic Fairness & Governance (Scaffolded & Ready)
- **Branch**: `phase-4/fairness-governance`
- **Scope**: Auditing algorithmic bias across sensitive protected attributes (gender, race, income), computing Disparate Impact, Demographic Parity, Equal Opportunity, and institutional AI governance policies.
- **Implementation**: Provided `PHASE4_FAIRNESS_GOVERNANCE_GUIDE.md` and `phase4_fairness_audit_pipeline.py`.

## Repository Hygiene & Systematic Branching
- **Branching Architecture**: Synchronized hierarchical phase branches (`phase-1/data-prep`, `phase-2/model-training`, `phase-3/explainable-ai`, `phase-4/fairness-governance`) to remote GitHub repository.
- **Deduplication**: Removed duplicate `requirements2.txt` file and standardized documentation paths.
