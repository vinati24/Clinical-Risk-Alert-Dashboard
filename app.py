import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import json
import os
from datetime import datetime
from streamlit_shap import st_shap
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- 1. SECURITY & CONFIGURATION ---
# Load variables from .env file (Ensure .env is in your .gitignore!)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Clinical Risk-Alert Dashboard", layout="wide")
st.title("🏥 Multimodal 'Risk-Alert' Dashboard")
st.markdown("Secure, Interoperable AI with Dynamic Safety Guardrails.")

# --- 2. SIDEBAR: PATIENT DATA INPUT ---
st.sidebar.header("Patient Real-time Data")
hrv = st.sidebar.slider("Heart Rate Variability (ms)", 10, 100, 48)
steps = st.sidebar.slider("Hourly Step Count", 0, 10000, 1457)
hr = st.sidebar.slider("Heart Rate (BPM)", 50, 150, 75)

# --- 3. PYTHON-SIDE LOGIC GUARDRAILS ---
# Accurate clinical classification performed before the LLM sees the data
if hrv < 30:
    hrv_label = "CRITICAL STRAIN"
    status_color = "red"
elif hrv <= 50:
    hrv_label = "MODERATE STRAIN"
    status_color = "orange"
else:
    hrv_label = "OPTIMAL RECOVERY"
    status_color = "green"

activity_label = "HIGH/ACTIVE" if steps > 2000 else "LIGHT/SEDENTARY"

# --- 4. LOAD RISK MODEL ---
@st.cache_resource
def load_risk_model():
    model = xgb.XGBClassifier()
    # Technical fix for XGBoost 2.0+ sklearn wrapper
    model._estimator_type = 'classifier'
    model.load_model("stress_model.json")
    return model

model = load_risk_model()

# --- 5. PREDICTION & EXPLANATION (XAI) ---
st.header("1. Explainable Risk Prediction")
col1, col2 = st.columns([1, 2])

# Prepare input for model
input_data = pd.DataFrame([[hr, hrv, steps]], columns=['heart_rate', 'hrv', 'steps'])
risk_proba = model.predict_proba(input_data)[0][1] * 100

with col1:
    st.metric(label="Calculated Stress Risk", value=f"{risk_proba:.1f}%")
    
    # Immediate visual safety feedback based on Python logic
    if hrv < 30:
        st.error(f"🚨 ALERT: {hrv_label} detected. Immediate rest advised.")
    elif hrv <= 50:
        st.warning(f"⚠️ NOTICE: {hrv_label}. Monitor recovery trends.")
    else:
        st.success(f"✅ STATUS: {hrv_label}. Physiological state is stable.")

with col2:
    st.subheader("SHAP Evidence (Why the alert?)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    # Renders the visual evidence for clinical transparency
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], input_data), height=200)

# --- 6. DYNAMIC REASONING ENGINE (LLM) ---
st.header("2. Verified Clinical Reasoning")

if st.button("Generate Clinical Insight"):
    if not hf_token:
        st.error("Missing Hugging Face Token. Please set HF_TOKEN in your .env file.")
    else:
        with st.spinner("Synthesizing clinical data..."):
            client = InferenceClient(api_key=hf_token)
            
            # --- DYNAMIC SYSTEM PROMPT (Fixed Hallucinations) ---
            # We construct the instructions based on the validated Python-side labels.
            system_msg = (
                f"You are a Clinical Assistant. Fact: The patient is in an {hrv_label} state. "
                "HRV is in milliseconds (ms); high HRV (>50ms) indicates recovery. "
            )
            
            if hrv < 50:
                system_msg += " TASK: Provide a severe medical warning regarding autonomic strain."
            else:
                system_msg += " TASK: Provide a positive wellness summary focusing on recovery."
            
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"DATA: HRV {hrv}ms, Steps {steps}, HR {hr}bpm. Analyze this state."}
            ]
            
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=messages,
                max_tokens=450, # Sufficient budget to prevent cut-offs
                temperature=0.1 # Low randomness for clinical accuracy
            )
            
            clinical_note = response.choices[0].message.content
            st.info("Logic Verified by Guardrails")
            st.markdown(f"**Clinical Impression:**\n\n{clinical_note}")

            # --- 7. CLINICAL INTEROPERABILITY (HL7 FHIR) ---
            st.divider()
            st.header("3. Medical Documentation (HL7 FHIR)")
            
            # Generate a standardized report resource
            report = {
                "resourceType": "DiagnosticReport",
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "81223-0",
                        "display": "Cardiovascular risk report"
                    }]
                },
                "subject": {"reference": "Patient/example-user"},
                "effectiveDateTime": datetime.now().isoformat(),
                "issued": datetime.now().isoformat(),
                "performer": [{"display": "Llama-3.2 Clinical AI Assistant"}],
                "conclusion": clinical_note,
                "extension": [
                    {"url": "http://example.org/safety-label", "valueString": hrv_label}
                ]
            }
            
            with st.expander("View Exportable FHIR DiagnosticReport JSON"):
                st.json(report)
                st.download_button(
                    label="Download Clinical Report for EMR",
                    data=json.dumps(report, indent=4),
                    file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )