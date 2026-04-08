import streamlit as st
import pandas as pd
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AFib Risk Prediction FL",
    page_icon="🫀",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data
def load_summary(file_path="project_summary.json"):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    st.title("🫀 Robust Federated Learning for Atrial Fibrillation")
    
    summary = load_summary()
    
    tab1, tab2 = st.tabs(["📊 Project Description", "🤖 AI Assistant (Placeholder)"])
    
    # --- TAB 1: PROJECT DESCRIPTION ---
    with tab1:
        if not summary:
            st.error("⚠️ `project_summary.json` not found. Please run the ML pipeline first.")
            return

        st.markdown("""
        ### 🎯 Project Objectives
        This project demonstrates a privacy-preserving Machine Learning pipeline designed to predict the risk of **Atrial Fibrillation (AFib)** within the first 24 hours of a patient's ICU admission. We compare a traditional centralized baseline against a robust Federated Learning (FL) approach designed to withstand adversarial attacks.
        """)
        
        st.divider()
        
        # --- ROW 1: Context ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🩺 Medical Context")
            st.markdown("""
            **What is Atrial Fibrillation (AFib)?**  
            AFib is an irregular and often very rapid heart rhythm (arrhythmia) that can lead to blood clots in the heart. It significantly increases the risk of stroke, heart failure, and other heart-related complications. 
            
            Early prediction in the ICU using continuous monitoring (vitals like heart rate, blood pressure) and laboratory results (like potassium levels) enables proactive clinical interventions.
            """)
            if "medical_context" in summary:
                st.info(f"**Clinical Focus:** {summary['medical_context']}")
            
        with col2:
            st.header("🗄️ Data Source (MIMIC)")
            data_ctx = summary.get("data_context", {})
            st.markdown(f"""
            We utilize the **MIMIC Database**, a large, freely-available database comprising de-identified health-related data associated with thousands of ICU admissions.
            
            *   **Total Patients:** `{data_ctx.get('total_patients', 'N/A')}`
            *   **AFib Prevalence:** `{data_ctx.get('af_prevalence_percent', 'N/A')}%`
            *   **Observation Window:** `{data_ctx.get('time_window', 'N/A')}`
            """)
            
        st.divider()
        
        # --- ROW 2: Architecture & Security ---
        st.header("🛡️ Why Federated Learning Matters")
        st.markdown("""
        Healthcare data is highly sensitive and strictly regulated (e.g., HIPAA). Centralizing data from multiple hospitals poses significant privacy and security risks. **Federated Learning (FL)** allows hospitals to collaboratively train a global ML model without ever sharing their raw patient data.
        """)
        
        arch_exp = summary.get("architecture_explanation", "N/A")
        st.warning(f"**Threat Model & Defense:** {arch_exp}")
        
        st.divider()
        
        # --- ROW 3: Final Results & Metrics ---
        st.header("📈 Final Results & Metrics")
        
        metrics = summary.get("metrics", {})
        xgb_metrics = metrics.get("centralized_xgboost", {"AUROC": 0.0, "F1": 0.0})
        fl_metrics = metrics.get("federated_robust_mlp", {"AUROC": 0.0, "F1": 0.0})
        
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline XGBoost (AUROC)", f"{xgb_metrics['AUROC']:.4f}")
        m2.metric("Federated MLP (AUROC)", f"{fl_metrics['AUROC']:.4f}", delta=f"{fl_metrics['AUROC'] - xgb_metrics['AUROC']:.4f}")
        m3.metric("Baseline XGBoost (F1)", f"{xgb_metrics['F1']:.4f}")
        m4.metric("Federated MLP (F1)", f"{fl_metrics['F1']:.4f}", delta=f"{fl_metrics['F1'] - xgb_metrics['F1']:.4f}")
        
        # Bar Chart
        st.subheader("Performance Comparison")
        
        chart_data = pd.DataFrame({
            "Model": ["Centralized XGBoost", "Robust Federated MLP"],
            "AUROC": [xgb_metrics["AUROC"], fl_metrics["AUROC"]],
            "F1 Score": [xgb_metrics["F1"], fl_metrics["F1"]]
        }).set_index("Model")
        
        st.bar_chart(chart_data)

    # --- TAB 2: AI ASSISTANT (PLACEHOLDER) ---
    with tab2:
        st.header("🤖 Clinical & Technical AI Assistant")
        st.info("The Ollama LLM integration is currently paused. Displaying placeholder auto-response.")
        
        with st.chat_message("user"):
            st.write("Can you show me the results from the latest pipeline run?")
            
        with st.chat_message("assistant"):
            st.write("Certainly! Here is the raw data collected from the federated learning simulation:")
            if summary:
                st.json(summary)
            else:
                st.error("No data available.")

if __name__ == "__main__":
    main()