import streamlit as st
import pandas as pd
import numpy as np
import predictor
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ICTREC Predictive Triage", page_icon="⚡", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def get_model():
    return predictor.load_pipeline()

pipeline = get_model()

if pipeline is None:
    st.error("Model not found! Please run 'python train_and_save.py' first.")
    st.stop()

# --- SIDEBAR (Controls & Context) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2082/2082875.png", width=80)
st.sidebar.title("Decision Controls")
escalation_threshold = st.sidebar.slider(
    "Escalation Threshold P(Delay)", 
    min_value=0.50, max_value=0.99, value=0.85, step=0.01,
    help="Tickets with a predicted delay probability above this threshold will trigger a Minute-1 Escalation."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**About this App**\n\nThis is a decision-support prototype built for the FUNAAB ICTREC Helpdesk. It uses a Hybrid NLP & XGBoost model to predict SLA breaches at minute-1.")

# --- TABS ---
tab1, tab2 = st.tabs(["🚀 Live Triage Simulator", "📊 Research & Business Impact"])

# --- TAB 1: LIVE SIMULATOR ---
with tab1:
    st.header("Live Triage Simulator")
    st.markdown("Test the predictive model in real-time. Use the presets below for a quick demonstration.")
    
    # Judge Presets
    colA, colB, colC = st.columns(3)
    preset = "None"
    if colA.button("🔄 Preset 1: The Baseline (Safe)"):
        preset = "Baseline"
    if colB.button("⚠️ Preset 2: The Deceptive Ticket (Danger)"):
        preset = "Deceptive"
    if colC.button("🧹 Clear Form"):
        preset = "None"

    # Set default values based on preset
    def_user = "Student" if preset != "Deceptive" else "Staff"
    def_dept = "Library" if preset != "Deceptive" else "ICTREC"
    def_cat = "Email" if preset != "Deceptive" else "Internet/Network"
    def_prio = "Low" if preset != "Deceptive" else "Low" # Intentionally deceptive Priority
    def_desc = "I forgot my email password and need a reset." if preset == "Baseline" else ("The main portal gateway is completely down and no one can process payments." if preset == "Deceptive" else "")

    # Layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Intake Form")
        user_type = st.selectbox("User Type", ["Student", "Postgraduate Student", "Staff"], index=["Student", "Postgraduate Student", "Staff"].index(def_user))
        unit_dept = st.selectbox("Department", ["Library", "Student Affairs", "Registry", "College of Animal Science", "Academic Planning", "ICTREC", "College of Veterinary Medicine", "Postgraduate School", "College of Engineering", "College of Plant Science", "College of Agricultural Management"], index=["Library", "Student Affairs", "Registry", "College of Animal Science", "Academic Planning", "ICTREC", "College of Veterinary Medicine", "Postgraduate School", "College of Engineering", "College of Plant Science", "College of Agricultural Management"].index(def_dept))
        issue_cat = st.selectbox("Issue Category", ["ID Card", "Registration", "Email", "Internet/Network", "Software/MIS", "Hardware", "Payment/Invoice", "Clearance", "Portal", "Admission"], index=["ID Card", "Registration", "Email", "Internet/Network", "Software/MIS", "Hardware", "Payment/Invoice", "Clearance", "Portal", "Admission"].index(def_cat))
        priority = st.selectbox("User Priority", ["Low", "Medium", "High", "Critical"], index=["Low", "Medium", "High", "Critical"].index(def_prio))
        issue_desc = st.text_area("Issue Description", value=def_desc, height=100)

    # Output area
    with col2:
        st.subheader("AI Triage Analysis")
        
        if issue_desc:
            # Predict
            prob = predictor.predict_ticket(pipeline, user_type, unit_dept, issue_cat, priority, issue_desc)
            
            # Display Gauge / Probability
            st.metric("Probability of Delay (>8 hrs)", f"{prob * 100:.1f}%")
            
            # Decision Logic
            if prob >= escalation_threshold:
                st.error(f"🚨 **MINUTE-1 ESCALATION TRIGGERED** 🚨\n\nProbability ({prob:.2f}) exceeds threshold ({escalation_threshold:.2f}). Bypassing Tier 1.")
            else:
                st.success(f"✅ **STANDARD QUEUE** ✅\n\nProbability ({prob:.2f}) is below threshold ({escalation_threshold:.2f}). Routing to Tier 1.")
            
            # Explainable AI (XAI) Feature
            trigger_words = predictor.explain_prediction(pipeline, issue_desc)
            if trigger_words:
                st.markdown("### 🔍 Explainability (Why?)")
                st.write("The NLP model detected the following high-risk keywords in the description:")
                # Highlight words using markdown tags
                for word in trigger_words:
                    st.markdown(f"- **`{word}`**")
        else:
            st.info("Enter an issue description to see the AI analysis.")

# --- TAB 2: RESEARCH & BUSINESS IMPACT ---
with tab2:
    st.header("Research Metrics & Business Impact")
    
    col_m1, col_m2 = st.columns(2)
    
    # Load Metrics
    try:
        metrics_df = pd.read_csv("app_data/evaluation_metrics.csv")
        cm_df = pd.read_csv("app_data/confusion_matrix.csv", index_col=0)
        
        with col_m1:
            st.subheader("Data Science Evaluation")
            st.dataframe(metrics_df.style.format({'Score': '{:.3f}'}), use_container_width=True)
            
            st.write("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)
            
        with col_m2:
            st.subheader("Executive Impact Calculator")
            st.markdown("Translating Model Metrics into Business Value (Simulated over 1,000 tickets).")
            
            # Simple business logic:
            # Every True Positive saves 7 hours of wasted wait time.
            # Every False Positive costs 1 hour of unnecessary senior dev triage.
            # From Confusion Matrix we can extract ratios (roughly)
            cm_vals = cm_df.values
            tn, fp, fn, tp = cm_vals.flatten()
            total_cases = np.sum(cm_vals)
            
            # Project to 1000 tickets
            multiplier = 1000 / total_cases
            proj_tp = tp * multiplier
            proj_fp = fp * multiplier
            
            hours_saved = proj_tp * 7 # Saved 7 hours of delay
            hours_wasted = proj_fp * 1 # Cost 1 hour
            net_hours = hours_saved - hours_wasted
            
            st.metric("Projected Escalations Correctly Handled", f"{int(proj_tp)}")
            st.metric("Projected SLA Breach Reduction Rate", f"{(tp/(tp+fn))*100:.1f}%")
            st.metric("Net Operational Hours Saved", f"{int(net_hours)} hrs", delta="Per 1k tickets")
            
            st.info("The Cost-Sensitive Learning approach explicitly prioritized recall, sacrificing overall accuracy to aggressively catch delayed tickets and protect SLAs.")

    except Exception as e:
        st.warning("Could not load research metrics. Please ensure 'train_and_save.py' was run.")
