# Clinical Risk-Alert Dashboard: Safe-by-Design LLM & Explainable Clinical Decision Support

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![XAI](https://img.shields.io/badge/XAI-SHAP-blueviolet.svg)](https://github.com/shap/shap)
[![FHIR](https://img.shields.io/badge/Interop-HL7%20FHIR-orange.svg)](https://www.hl7.org/fhir/)
[![LLM](https://img.shields.io/badge/LLM-Llama--3.2-green.svg)](https://huggingface.co/meta-llama)

## Research Context

Large Language Models (LLMs) are increasingly used to generate clinical documentation, but small-parameter models (≤ 3B) are prone to **Numerical Grounding Failures** — misinterpreting critical physiological values as benign. This project develops a Clinical Decision Support System (CDSS) that integrates predictive ML, explainable AI, and LLM-powered note generation with a **dual-layer safety guardrailing architecture** that ensures 100% logical consistency between numerical predictions and generated clinical text.

The system also demonstrates end-to-end clinical data interoperability by mapping outputs to **HL7 FHIR DiagnosticReport** resources, and privacy-preserving development using **CTGAN**-synthesized patient data.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PATIENT INPUT LAYER                            │
│        Wearable Sensor Data: HRV (ms) · Heart Rate · Steps        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                   PREDICTION ENGINE                                │
│                                                                    │
│  ┌──────────────────┐    ┌────────────────────────────────┐       │
│  │  XGBoost          │    │  SHAP TreeExplainer            │       │
│  │  Classifier       │───▶│  (Force plots for clinical     │       │
│  │  (Stress Risk %)  │    │   transparency)                │       │
│  └────────┬─────────┘    └────────────────────────────────┘       │
└───────────┼─────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────────────┐
│              DUAL-LAYER SAFETY GUARDRAIL SYSTEM                    │
│                                                                    │
│  ┌─────────────────────────────────────────────┐                  │
│  │  LAYER 1: Python Logic Guardrails           │                  │
│  │  Hard-coded medical reference ranges         │                  │
│  │  HRV < 30ms → CRITICAL STRAIN               │                  │
│  │  HRV 30-50ms → MODERATE STRAIN              │                  │
│  │  HRV > 50ms → OPTIMAL RECOVERY              │                  │
│  └──────────────────┬──────────────────────────┘                  │
│                     │ Validated status passed to LLM               │
│  ┌──────────────────▼──────────────────────────┐                  │
│  │  LAYER 2: Dynamic System Prompting          │                  │
│  │  Constrains Llama-3.2-1B to generate notes  │                  │
│  │  consistent with Layer 1 classification      │                  │
│  │  Temperature: 0.1 (minimal randomness)       │                  │
│  └──────────────────┬──────────────────────────┘                  │
└─────────────────────┼───────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│              CLINICAL INTEROPERABILITY LAYER                       │
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  HL7 FHIR DiagnosticReport (LOINC 81223-0)          │         │
│  │  Exportable JSON for EHR/EMR integration             │         │
│  └──────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 🔬 Predictive Analytics
- **XGBoost Classifier** trained on multimodal wearable sensor features (HRV, Heart Rate, Step Count)
- Real-time stress risk probability with configurable thresholds

### 🧠 Explainable AI (XAI)
- **SHAP force plots** rendered in-dashboard for every prediction
- Clinicians see *which features drove the alert* — not just the alert itself

### 🛡️ AI Safety: Dual-Layer Guardrailing
The core research contribution. During development, I identified that Llama-3.2-1B suffers from **Numerical Grounding Failures** — interpreting `HRV = 16ms` (critical autonomic strain) as "optimal recovery." The dual-layer system:

1. **Layer 1 (Python Logic)**: Performs clinical classification using hard-coded medical reference ranges *before* the LLM is invoked
2. **Layer 2 (Dynamic Prompting)**: Dynamically adjusts the LLM's system instructions based on the validated status, preventing *Instruction Contamination*

**Result**: 100% logical consistency across all clinical safety audit test cases, including adversarial "Metric Flip" scenarios.

### 🏥 Clinical Interoperability
- Automated mapping to **HL7 FHIR DiagnosticReport** resources (LOINC 81223-0)
- Downloadable JSON for EHR/EMR integration
- See [`docs/example_fhir_observation.json`](docs/example_fhir_observation.json) for a sample resource

### 🔒 Privacy-Preserving Development
- 10,000+ synthetic patient records generated via **CTGAN**
- Distributional fidelity validated against source distributions
- Full data generation pipeline in [`notebooks/data_generation.ipynb`](notebooks/data_generation.ipynb)

## Project Structure

```
Clinical-Risk-Alert-Dashboard/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── app.py                          # Streamlit application entry point
├── models/
│   └── stress_model.json           # Trained XGBoost model
├── data/
│   ├── synthetic_health_data.csv           # Full synthetic dataset (CTGAN)
│   └── synthetic_health_data_anonymized.csv  # Anonymized subset
├── notebooks/
│   └── data_generation.ipynb       # CTGAN data synthesis pipeline
└── docs/
    └── example_fhir_observation.json  # Sample HL7 FHIR resource
```

## Getting Started

### Prerequisites
- Python 3.8+
- A [Hugging Face API token](https://huggingface.co/settings/tokens) (for Llama-3.2-1B inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/vinati24/Clinical-Risk-Alert-Dashboard.git
cd Clinical-Risk-Alert-Dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`. Use the sidebar sliders to input patient vital signs and generate predictions.

## Validation & Safety Audit

| Test Case | HRV | Steps | Expected Status | LLM Output | Pass |
|-----------|-----|-------|-----------------|------------|------|
| Critical Strain | 16ms | 500 | ⛔ CRITICAL | Warning issued | ✅ |
| Moderate Strain | 45ms | 1200 | ⚠️ MODERATE | Monitoring advised | ✅ |
| Optimal Recovery | 72ms | 3000 | ✅ OPTIMAL | Positive summary | ✅ |
| Metric Flip (adversarial) | 75ms | 8000 | ✅ OPTIMAL | Positive summary | ✅ |

**100% logical consistency** across all audit test cases.

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Prediction | XGBoost | Physiological strain classification |
| Explainability | SHAP | Feature importance & force plots |
| LLM | Llama-3.2-1B (via HuggingFace) | Clinical note synthesis |
| Safety | Custom Python guardrails | Hallucination prevention |
| Interoperability | HL7 FHIR (LOINC 81223-0) | EHR/EMR integration |
| Synthetic Data | CTGAN | Privacy-preserving training data |
| Frontend | Streamlit | Interactive clinical dashboard |

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{nathwani2025clinicalrisk,
  author = {Nathwani, Vinati},
  title = {Clinical Risk-Alert Dashboard: Safe-by-Design LLM and Explainable Clinical Decision Support},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/vinati24/Clinical-Risk-Alert-Dashboard}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Vinati Nathwani**
- MSc AI for Biomedicine and Healthcare — University College London (UCL)
- BTech Computer Science (Health Informatics) — VIT Bhopal
- [GitHub](https://github.com/vinati24) · [LinkedIn](https://linkedin.com/in/vinati-nathwani-42b622260)
