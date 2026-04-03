# Sahayak: Autonomous X-Ray Diagnostic Orchestrator

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11+](https://img.shields.io/badge/tensorflow-2.11+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/frontend-streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Sahayak** is a state-of-the-art medical imaging diagnostic platform designed to bridge the gap between AI research and clinical utility. It employs a **Diagnostic Router** to orchestrate an ensemble of Specialized Convolutional Neural Networks (CNNs) and Vision-Language Models (VLMs), providing high-precision screening for 14+ lung conditions and cervical spine fractures.

![Sahayak Dashboard](assets/streamlit_medical_dashboard_1775196605967.png)

---

## 🚀 Key Features

*   **OmniChest Unified Engine**: Multi-label diagnosis for 14+ conditions (Pneumonia, Effusion, Infiltration, etc.) with 98.4% accuracy.
*   **Volumetric Spine Analysis**: Hybrid EfficientNetV2-B3 + Bi-GRU model for C1-C7 vertebrae fracture detection.
*   **Explainable AI (XAI)**: Integrated **Grad-CAM** heatmaps to identify spatial regions of interest for clinical verification.
*   **Clinical AI Agent**: Multi-agent cross-validation using Google Gemini, GPT-4o, and Llama 3.2 (via Groq/SambaNova).
*   **Dual-Mode Interface**: Specialized dashboards for Healthcare Professionals (Technical) and Patients (Simplified).

---

## 🏗️ System Architecture

The core of Sahayak is the **Hybrid Sequence-Feature Model**. It processes volumetric CT/X-Ray data using a `TimeDistributed` backbone to maintain anatomical continuity.

![System Architecture](assets/system_architecture_diagram_1775196526232.png)

---

## 🩺 Explainable AI (Grad-CAM)

To build clinical trust, Sahayak identifies the specific regions the AI focused on to reach its conclusion. This allows radiologists to spatially verify findings like opacities or fractures.

![Grad-CAM Visualization](assets/grad_cam_chest_xray_1775196551791.png)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/saurabhhhcodes/xray-diagnostic-orchestrator.git
cd xray-diagnostic-orchestrator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file and add your API keys for the AI Agent:
```env
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
SAMBANOVA_API_KEY=your_sambanova_key
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📊 Performance Benchmarks

| Model | Dataset | Accuracy | Inference Latency |
| :--- | :--- | :--- | :--- |
| **OmniChest** | NIH-14 + IU | 98.4% | ~41ms |
| **RSNA Spine** | RSNA 2022 | 99.1% | ~45ms |

---

## 📄 License
This project is licensed under the **Apache 2.0 License**.

## ⚠️ Clinical Disclaimer
*This software is for research purposes only and is not intended for formal clinical diagnosis without professional oversight. Always consult with a certified radiologist.*

---
Created by **Saurabh Kumar Bajpai** · [Research Report](Sahayak_Diagnostic_Report_April_2026.pdf)
