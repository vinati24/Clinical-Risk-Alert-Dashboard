# Multimodal Clinical Risk-Alert Dashboard
### *Explainable AI (XAI) & Hallucination-Proof Medical Reasoning*

## Overview
This project is a clinical decision support system designed to predict physiological stress events using wearable data such as HRV, Heart Rate, and Step Count. It features a custom Safety Guardrail Layer that eliminates LLM reasoning drift, ensuring that AI-generated clinical impressions remain grounded in medical reality.

## Technical Architecture
* **Predictive Engine:** A high-performance **XGBoost Classifier** trained to identify physiological strain patterns.
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to solve the "Black Box" transparency problem, providing clinicians with visual evidence for every prediction.
* **Generative Reasoning:** Leverages **Llama-3.2-1B** to synthesize complex data into structured, professional clinical notes.
* **Interoperability:** Automated mapping to **HL7 FHIR DiagnosticReport** resources (LOINC 81223-0) for seamless EMR integration.
* **Privacy-Preserving Data:** Training data synthesized via **CTGAN** to ensure HIPAA-compliant development workflows.

## AI Safety: Solving the Hallucination Problem
During development, I identified that small-parameter models (1B) often suffer from Numerical Grounding Failures , misinterpreting critical strain (e.g., 16ms HRV) as "optimal recovery".

### The Solution: Dynamic Logic Guardrails
I implemented a dual-layer validation system to ensure 100% logical consistency:
1.  **Python Logic Layer:** Performs clinical categorization using hard-coded medical reference ranges before the LLM is invoked.
2.  **Dynamic System Prompting:** The system dynamically adjusts the LLM's instructions based on the Python-validated status. This prevents **Instruction Contamination**, ensuring the model only issues warnings when a true physiological risk is detected.

## 📊 Performance & Validation
* **Statistical Drivers:** SHAP analysis identified Step Count and HRV as the primary drivers of stress alerts, aligning with clinical literature on autonomic strain.
* **Safety Audit:** Verified logical consistency across "Metric Flip" test cases (e.g., High Activity + High HRV correctly results in an "Optimal Recovery" status).

---
*Developed by a Health-AI Engineer focused on Responsible AI and Clinical Interoperability.*
