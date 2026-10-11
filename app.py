"""
Unified AI Governance & Student Dropout Retention Platform
Built with Streamlit
Replication & Institutional Extension of IEEE RITA 2025 (Silva et al.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════
# Page Config & Custom Styling
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Dropout XAI & Governance Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-pass {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warn {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# Helper Data Loaders (Cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_audit_data():
    ew_df = pd.read_csv("phase4_outputs/early_warning_vs_postmortem_audit.csv") if os.path.exists("phase4_outputs/early_warning_vs_postmortem_audit.csv") else pd.DataFrame()
    mod_df = pd.read_csv("phase4_outputs/modality_disaggregated_audit.csv") if os.path.exists("phase4_outputs/modality_disaggregated_audit.csv") else pd.DataFrame()
    fair_df = pd.read_csv("phase4_outputs/fairness_metrics_summary.csv") if os.path.exists("phase4_outputs/fairness_metrics_summary.csv") else pd.DataFrame()
    inter_df = pd.read_csv("phase4_outputs/intersectional_fairness_summary.csv") if os.path.exists("phase4_outputs/intersectional_fairness_summary.csv") else pd.DataFrame()
    exp_df = pd.read_csv("experiment_results.csv") if os.path.exists("experiment_results.csv") else pd.DataFrame()
    shap_df = pd.read_csv("phase3_outputs/shap_feature_attributions.csv") if os.path.exists("phase3_outputs/shap_feature_attributions.csv") else pd.DataFrame()
    
    with open("phase4_outputs/interactive_students_xai.json", "r", encoding="utf-8") as f:
        students_xai = json.load(f)
    with open("phase4_outputs/counterfactual_curves.json", "r", encoding="utf-8") as f:
        cf_curves = json.load(f)
    with open("phase4_outputs/decoded_student_metadata.json", "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    return ew_df, mod_df, fair_df, inter_df, exp_df, shap_df, students_xai, cf_curves, meta_dict

@st.cache_resource
def load_model_bundle():
    if os.path.exists("phase3_bundle.pkl"):
        return joblib.load("phase3_bundle.pkl")
    return None

ew_df, mod_df, fair_df, inter_df, exp_df, shap_df, students_xai, cf_curves, meta_dict = load_audit_data()
bundle = load_model_bundle()

# ═══════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
st.sidebar.markdown("### **AI Governance Portal**")
st.sidebar.caption("Brazilian Technical Education Attrition Prevention")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Sections:",
    [
        "🏆 Phase 2: Champion Benchmark & Validation",
        "🌐 Phase 3: Macro Explainability (Global SHAP)",
        "🎛️ Phase 3: Micro XAI & What-If Simulator",
        "⚖️ Phase 4: Algorithmic Fairness & Intersectional Matrix",
        "📜 Phase 4: Ethical AI Governance Charter"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Champion Model**: Random Forest  
**Recall**: 84.8% | **Precision**: 83.6%  
**Accuracy**: 92.9% | **ROC-AUC**: 0.951  
**Test Cohort**: 3,017 students
""")

# ═══════════════════════════════════════════════════════════════════════
# Header Banner
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🎓 Student Retention XAI & Algorithmic Governance Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Replication & Comprehensive Extension of Silva et al. (IEEE RITA 2025) across 15,084 Vocational Students</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Champion Benchmark & Real-World Validation
# ═══════════════════════════════════════════════════════════════════════
if nav_choice == "🏆 Phase 2: Champion Benchmark & Validation":
    st.header("Phase 2: 48-Experiment Model Benchmark & Validation")
    
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Champion Recall", "84.8%", "Caught 573/676 dropouts")
    k2.metric("Champion Precision", "83.6%", "Only 112 false alarms")
    k3.metric("Overall Accuracy", "92.9%", "+7.7% over base paper")
    k4.metric("ROC-AUC Score", "0.951", "Excellent discrimination")

    st.markdown("---")
    
    st.subheader("1. Real-World Early-Warning Validation (GPA > 0 vs. GPA == 0)")
    st.markdown("""
    **Core Research Soundness**: In school registration databases, students who abandon their studies on Day 1 receive an administrative `GPA = 0.0`. 
    To prove our model is a genuine early-warning system and does not rely on trivial zero-GPA signals, we audited active enrolled students separately:
    """)
    st.dataframe(ew_df, use_container_width=True)

    st.success("""
    ✓ **Empirical Proof**: On active students who attended classes and earned grades (`GPA > 0`), the model maintains **80.8% Recall and 82.5% Precision**, confirming high clinical predictive validity for early semester interventions.
    """)

    st.markdown("---")
    st.subheader("2. Course Modality Disaggregation Audit")
    st.markdown("""
    Different vocational modalities possess drastically different baseline attrition rates. Adult evening programs (`Subsequente` and `EJA`) experience over 2x higher real-world dropout rates than teen high schoolers (`Integrado`):
    """)
    st.dataframe(mod_df, use_container_width=True)

    st.info("""
    💡 **Insight**: High flagging rates for adult students reflect real-world socioeconomic pressures (balancing full-time employment with evening study), backed by high precision (90.4%–93.5%).
    """)

    st.markdown("---")
    st.subheader("3. Top Benchmark Models (Out of 48 Experiments)")
    if not exp_df.empty:
        # Display sorted by recall then precision
        df_sorted = exp_df.sort_values(by=["Recall", "Precision"], ascending=[False, False]).head(10)
        st.dataframe(df_sorted[["Model", "Balancing", "Feature_Selection", "Recall", "Precision", "Accuracy", "ROC_AUC"]], use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Macro Explainability (Global SHAP)
# ═══════════════════════════════════════════════════════════════════════
elif nav_choice == "🌐 Phase 3: Macro Explainability (Global SHAP)":
    st.header("Phase 3: Macro Global Explainability via TreeExplainer SHAP")
    st.markdown("""
    Global feature attributions explain the macro logic of the Random Forest across all **3,017 test students**. 
    All feature names are fully translated and mapped to intuitive English educational concepts.
    """)

    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        st.subheader("Global SHAP Beeswarm Attributions")
        if os.path.exists("phase3_outputs/01_shap_global_beeswarm.png"):
            st.image("phase3_outputs/01_shap_global_beeswarm.png", use_container_width=True, caption="Figure 1: SHAP Beeswarm Plot (Higher SHAP value = Increased Dropout Risk)")
    
    with c2:
        st.subheader("Feature Importance Table")
        st.dataframe(shap_df, use_container_width=True)
        st.markdown("""
        **Top Decisive Predictors**:
        1. **Cumulative GPA (`ira`)**: Decisive academic anchor. Dropping below 5.0 massively escalates risk.
        2. **Attendance Rate (`percentual_frequencia`)**: Primary behavioral indicator of institutional disengagement.
        3. **Failed Courses (`reprovacoes`)**: Compounding academic struggle.
        4. **Admission Year (`anoingresso`)**: Cohort & curriculum transition factors.
        5. **Gross Family Income (`rendabruta`)**: Financial vulnerability buffer.
        """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Micro XAI & Interactive What-If Simulator
# ═══════════════════════════════════════════════════════════════════════
elif nav_choice == "🎛️ Phase 3: Micro XAI & What-If Simulator":
    st.header("Phase 3: Micro XAI & Interactive Counterfactual Simulator")
    st.markdown("Explore individual student profiles, view local LIME factors, and simulate remedial interventions in real time.")

    # Archetype selector
    student_keys = list(students_xai.keys())
    student_labels = [f"Student #{k} — Risk: {students_xai[k]['risk_score']*100:.1f}% ({students_xai[k]['risk_category']})" for k in student_keys]
    
    selected_label = st.selectbox("Select Student Profile Archetype:", student_labels)
    selected_idx = student_keys[student_labels.index(selected_label)]
    
    curr_student = students_xai[selected_idx]
    curr_meta = meta_dict.get(str(selected_idx), {})
    curr_cf = cf_curves.get(str(selected_idx), {})

    col_meta, col_lime, col_sim = st.columns([1, 1.2, 1.3])

    with col_meta:
        st.subheader("Student Profile")
        st.markdown(f"""
        - **Profile**: Student #{selected_idx}
        - **Predicted Risk**: **{curr_student['risk_score']*100:.1f}%**
        - **Risk Level**: `{curr_student['risk_category']}`
        - **Cumulative GPA**: `{curr_meta.get('Cumulative GPA (ira)', 'N/A')}`
        - **Attendance**: `{curr_meta.get('Attendance Rate (%)', 'N/A')}`
        - **Failed Courses**: `{curr_meta.get('Failed Courses (reprovacoes)', 'N/A')}`
        - **Modality**: `{curr_meta.get('Course Modality', 'N/A')}`
        - **Household Income**: `{curr_meta.get('Gross Family Income (R$)', 'N/A')}`
        - **Laptops Owned**: `{curr_meta.get('Laptops Owned', 'N/A')}`
        - **Living With**: `{curr_meta.get('Living Arrangement', 'N/A')}`
        """)

    with col_lime:
        st.subheader("LIME Factor Attributions")
        factors = curr_student.get("lime_factors", [])
        if factors:
            f_df = pd.DataFrame(factors)
            f_df["Type"] = f_df["direction"].apply(lambda x: "Risk Factor" if x == "risk" else "Protective Buffer")
            
            fig = px.bar(
                f_df, 
                x="weight", 
                y="feature", 
                orientation="h",
                color="Type",
                color_discrete_map={"Risk Factor": "#ef4444", "Protective Buffer": "#10b981"},
                title="Local Impact on Risk Score"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=35, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)

    with col_sim:
        st.subheader("Intervention Simulator")
        st.markdown("Adjust key variables to see how counseling/tutoring alters the risk:")
        
        sim_att = st.slider("Simulated Attendance Rate (%)", min_value=30, max_value=100, value=75, step=5)
        sim_gpa = st.slider("Simulated Cumulative GPA (0-10)", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

        # Estimate simulated risk from precomputed curves
        att_curve = curr_cf.get("attendance", [])
        gpa_curve = curr_cf.get("gpa", [])
        
        # Approximate combined risk
        base_risk = curr_student["risk_score"]
        # Find closest attendance risk
        att_risk = base_risk
        for pt in att_curve:
            if pt["attendance"] <= sim_att:
                att_risk = pt["risk"]
        
        gpa_risk = base_risk
        for pt in gpa_curve:
            if pt["gpa"] <= sim_gpa:
                gpa_risk = pt["risk"]
        
        sim_est_risk = (att_risk + gpa_risk) / 2.0

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sim_est_risk * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Simulated Dropout Risk (%)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f2937"},
                'steps': [
                    {'range': [0, 40], 'color': "#86efac"},
                    {'range': [40, 65], 'color': "#fde047"},
                    {'range': [65, 100], 'color': "#fca5a5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if sim_est_risk < 0.40:
            st.success("Target Outcome: Safe Graduation Zone. Continue regular academic check-ins.")
        elif sim_est_risk < 0.65:
            st.warning("Target Outcome: Moderate Risk. Recommend peer tutoring and weekly attendance check.")
        else:
            st.error("Target Outcome: Critical High Risk. Immediate institutional counseling & academic agreement required.")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Algorithmic Fairness & Intersectional Matrix
# ═══════════════════════════════════════════════════════════════════════
elif nav_choice == "⚖️ Phase 4: Algorithmic Fairness & Intersectional Matrix":
    st.header("Phase 4: Algorithmic Fairness & Intersectional Equity Auditing")
    st.markdown("""
    Comprehensive equity audit across demographic and socioeconomic slices adhering to the **US EEOC Four-Fifths (80%) Rule** 
    and **Equal Opportunity Parity** on **100% real empirical student records**.
    """)

    st.subheader("1. Single-Dimension Fairness Metrics")
    st.dataframe(fair_df, use_container_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("""
        **Compliance Findings**:
        - **Race / Ethnicity (White vs. Minority)**: **Disparate Impact = 1.180 (PASSED - Fair)**.
        - **Gender (Male vs. Female)**: Ratio = 0.707 (Reflects higher actual male attrition in vocational engineering tracks).
        - **Household Income & Age**: Flags low-income and adult students slightly more often because their actual baseline dropout rates are 2x higher.
        """)
    with col_f2:
        if os.path.exists("phase4_outputs/01_comprehensive_fairness_audit.png"):
            st.image("phase4_outputs/01_comprehensive_fairness_audit.png", use_container_width=True, caption="Figure 2: 4-Panel Quantitative Fairness Audit")

    st.markdown("---")
    st.subheader("2. Compound Intersectional Matrix (Gender × Race × Income)")
    st.markdown("""
    Evaluating compound vulnerability slices on **100% real student data (no synthetic profiles)**:
    """)
    st.dataframe(inter_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
# TAB 5: Ethical AI Governance Charter
# ═══════════════════════════════════════════════════════════════════════
elif nav_choice == "📜 Phase 4: Ethical AI Governance Charter":
    st.header("Phase 4: Institutional AI Governance Charter & Safeguards")
    st.markdown("""
    Binding operational policies governing the clinical, advisory deployment of machine learning in public technical education.
    """)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        ### Pillar 1: Strictly Advisory Risk Indicator
        - Automated academic tracking, disciplinary measures, or enrollment dismissal based on model predictions are **strictly prohibited**.
        - Risk scores serve exclusively as an early-warning prioritization dashboard for certified educational counselors.

        ### Pillar 2: Explainability-Coupled Outreach
        - Counselors are prohibited from contacting students with generic risk alerts.
        - Every outreach must present the primary SHAP and LIME risk drivers so intervention directly addresses the root cause (e.g. food subsidies vs. math tutoring).

        ### Pillar 3: Continuous Bias & Drift Auditing
        - Bi-annual re-evaluations of Disparate Impact and Equal Opportunity are required before each academic intake.
        - Decision thresholds must be recalibrated if demographic distributions shift by >5%.
        """)

    with p2:
        st.markdown("""
        ### Pillar 4: Independent Student Right to Explanation & Appeal
        - Any student receiving secondary academic intervention maintains the institutional right to review factor attributions with guidance staff.
        - Students have the formal right to request manual counselor re-evaluation.

        ### Pillar 5: Certified Human-in-the-Loop Override
        - Educational counselors and faculty committees retain 100% final override authority over all algorithmic recommendations.
        - Human clinical assessments supersede algorithmic risk classifications.
        """)

    st.success("Certified Institutional Framework: Compliant with UNESCO & IEEE Educational AI Ethics Recommendations (2025).")
