import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import matplotlib.cm as cm
# from xray_service import NIHPredictor, RSNAPredictor, PadChestPredictor # Removed

from ai_agent import analyze_xray_with_agent, get_agent_findings_summary

st.set_page_config(page_title="Pro X-Ray Diagnostic", page_icon="🩻", layout="wide")

# --- CSS for "Production" Feel ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #2c3e50;
    }
    .badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .badge-success { background-color: #d4edda; color: #155724; }
    .badge-warning { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# --- MEDICAL KNOWLEDGE BASE ---
# --- MEDICAL KNOWLEDGE BASE ---
DISEASE_DETAILS = {
    'Atelectasis': "Partial collapse or incomplete expansion of the lung.",
    'Cardiomegaly': "Enlarged heart, often a sign of other heart conditions.",
    'Consolidation': "A region of lung tissue that has filled with liquid instead of air.",
    'Edema': "Abnormal accumulation of fluid in the lung tissues.",
    'Effusion': "Buildup of fluid between the layers of tissue that line the lungs and chest cavity.",
    'Emphysema': "Damage to the air sacs (alveoli) in the lungs, causing shortness of breath.",
    'Fibrosis': "Scarring of the lung tissue, which can lead to breathing problems.",
    'Hernia': "An organ squeezes through a weak spot in a surrounding muscle or tissue.",
    'Infiltration': "A substance (like pus, blood, or protein) lingering in the lung tissue.",
    'Mass': "A large abnormal growth or lesion (>3cm), potential tumor.",
    'No Finding': "No obvious abnormalities were detected by the model.",
    'Nodule': "A small abnormal growth or lump (<3cm).",
    'Pleural_Thickening': "Thickening or scarring of the lining of the lungs (pleura).",
    'Pneumonia': "Infection that inflames the air sacs in one or both lungs.",
    'Pneumothorax': "A collapsed lung caused by air leaking into the space between the lung and chest wall.",
    'Normal': "No significant abnormalities detected."
}

PATIENT_FRIENDLY_DESCRIPTIONS = {
    'Atelectasis': "The lung isn't inflating fully. This can happen after surgery or from blockage.",
    'Cardiomegaly': "The heart looks larger than usual. Your doctor may check your blood pressure or heart health.",
    'Consolidation': "Part of the lung is filled with fluid, often from an infection like pneumonia.",
    'Edema': "There is some fluid in the lungs, which can make breathing harder.",
    'Effusion': "Fluid has collected around the lungs. This can cause shortness of breath.",
    'Emphysema': "The air sacs are damaged, often related to smoking or long-term exposure.",
    'Fibrosis': "Some scarring was found in the tissue, which might be from an old infection.",
    'Hernia': "Part of the stomach or intestine has pushed up into the chest area.",
    'Infiltration': "There are signs of something (like infection) spreading in the lung tissue.",
    'Mass': "A larger spot (>3cm) was found that needs careful checking by a doctor.",
    'No Finding': "Good news! We didn't find any obvious signs of disease.",
    'Nodule': "A small spot (<3cm) was seen. These are often benign but should be tracked.",
    'Pleural_Thickening': "The lining of the lungs looks a bit thicker than normal.",
    'Pneumonia': "There are signs of a lung infection. You might need antibiotics.",
    'Pneumothorax': "Air has leaked out of the lung, which can cause it to collapse partially.",
    'Normal': "Good news! Your X-ray looks healthy."
}

def overlay_heatmap(img_path, heatmap):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap
    # Fixed deprecation: cm.get_cmap("jet") -> matplotlib.colormaps["jet"]
    try:
        jet = cm.get_cmap("jet")
    except:
        jet = matplotlib.colormaps["jet"] if hasattr(matplotlib, 'colormaps') else cm.Greys_r
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))
    return superimposed_img

# --- RISK STRATIFICATION ---
DISEASE_SEVERITY = {
    'Pneumothorax': 'critical', 'Pneumonia': 'critical', 'Effusion': 'critical',
    'Mass': 'significant', 'Nodule': 'significant', 'Consolidation': 'significant',
    'Infiltration': 'significant', 'Edema': 'significant',
    'Cardiomegaly': 'moderate', 'Atelectasis': 'moderate', 'Fibrosis': 'moderate',
    'Pleural_Thickening': 'minor', 'Emphysema': 'minor', 'Hernia': 'minor',
    'No Finding': 'normal', 'Normal': 'normal'
}

def calculate_risk_assessment(predictions):
    critical_findings, significant_findings, moderate_findings = [], [], []
    for disease, score in predictions.items():
        severity = DISEASE_SEVERITY.get(disease, 'minor')
        if score > 0.5 and severity == 'critical':
            critical_findings.append((disease, score))
        elif score > 0.4 and severity == 'significant':
            significant_findings.append((disease, score))
        elif score > 0.3 and severity == 'moderate':
            moderate_findings.append((disease, score))
    if critical_findings:
        return "CRITICAL", critical_findings
    elif significant_findings:
        return "HIGH", significant_findings
    elif moderate_findings:
        return "MODERATE", moderate_findings
    return "LOW", []

def display_comprehensive_findings(predictions, mode="professional"):
    risk_level, key_findings = calculate_risk_assessment(predictions)
    sorted_preds = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
    
    if mode == "professional":
        st.markdown("### 📊 Comprehensive Analysis")
        risk_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
        st.markdown(f"**Overall Risk**: {risk_colors.get(risk_level, '⚪')} **{risk_level}**")
        if key_findings:
            st.markdown("**Key Findings:**")
            for disease, score in key_findings:
                st.markdown(f"- **{disease}**: {score*100:.1f}%")
        st.markdown("---")
        st.markdown("### All Detected Conditions")
        critical, significant, moderate, minor = [], [], [], []
        for disease, score in sorted_preds.items():
            severity = DISEASE_SEVERITY.get(disease, 'minor')
            if severity == 'critical' and score > 0.1:
                critical.append((disease, score))
            elif severity == 'significant' and score > 0.1:
                significant.append((disease, score))
            elif severity == 'moderate' and score > 0.1:
                moderate.append((disease, score))
            elif score > 0.05:
                minor.append((disease, score))
        if critical:
            st.markdown("#### 🔴 Critical Findings")
            for disease, score in critical:
                with st.expander(f"{disease}: {score*100:.1f}%", expanded=(score > 0.5)):
                    st.progress(score)
                    st.caption(DISEASE_DETAILS.get(disease, ""))
        if significant:
            st.markdown("#### 🟠 Significant Findings")
            for disease, score in significant:
                with st.expander(f"{disease}: {score*100:.1f}%", expanded=(score > 0.5)):
                    st.progress(score)
                    st.caption(DISEASE_DETAILS.get(disease, ""))
        if moderate:
            st.markdown("#### 🟡 Moderate Findings")
            for disease, score in moderate:
                with st.expander(f"{disease}: {score*100:.1f}%"):
                    st.progress(score)
                    st.caption(DISEASE_DETAILS.get(disease, ""))
        if minor:
            with st.expander("📋 Other Observations"):
                for disease, score in minor:
                    st.markdown(f"- **{disease}**: {score*100:.1f}%")
    else:
        top_disease = list(sorted_preds.keys())[0]
        top_score = sorted_preds[top_disease]
        if top_disease in ["Normal", "No Finding"] and top_score > 0.5:
            st.balloons()
            st.success("### ✅ Likely Healthy")
            st.write(PATIENT_FRIENDLY_DESCRIPTIONS.get("Normal"))
        else:
            if risk_level == "CRITICAL":
                st.error(f"### 🚨 Critical: {top_disease}")
            elif risk_level == "HIGH":
                st.warning(f"### ⚠️ Attention Needed: {top_disease}")
            else:
                st.info(f"### ℹ️ Finding: {top_disease}")
            st.write(PATIENT_FRIENDLY_DESCRIPTIONS.get(top_disease, "Please consult a doctor."))
            st.progress(top_score)
            if key_findings and len(key_findings) > 1:
                st.markdown("**Additional findings detected:**")
                for disease, score in key_findings[1:3]:
                    st.markdown(f"- {disease} ({score*100:.0f}%)")

from xray_service import DiagnosticRouter

st.title("🩻 Sahayak Diagnostic Orchestrator")

# --- USER MODE SELECTION ---
user_mode = st.sidebar.radio("User Mode", ["Patient", "Healthcare Professional"])

# --- ROUTER CONFIGURATION ---
st.sidebar.title("⚙️ Analysis Settings")
body_part = st.sidebar.selectbox(
    "Select Body Part / Scan Type",
    ["Chest X-Ray", "Brain MRI (Coming Soon)", "Cervical Spine Scan", "Retinal Scan (Coming Soon)"]
)

engine_mode = st.sidebar.radio(
    "AI Engine",
    ["Legacy Ensemble (Recommended)", "Unified SOTA (Pneumonia Only)"],
    help="Legacy: Comprehensive 15-disease detection. Unified: Binary Pneumonia classifier with metadata."
)

# Metadata for Unified Model
meta_data = {}
if "Unified" in engine_mode:
    st.sidebar.markdown("### Patient Data")
    age = st.sidebar.number_input("Patient Age", min_value=0, max_value=100, value=50)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    meta_data = {"age": age, "gender": gender}


# Initialize Router
@st.cache_resource
def get_router():
    return DiagnosticRouter()

router = get_router()

if user_mode == "Healthcare Professional":
    st.markdown(f"**active specialist:** *{body_part} Ensemble*")
    st.info("The **Diagnostic Router** is active. It will automatically ensemble multiple models to reduce false positives.")
else:
    st.markdown("**Smart Health Assistant** - We automatically choose the best specialist for you.")


uploaded_file = None

if user_mode == "Patient":
    input_method = st.radio("How would you like to provide the Scan?", ["Upload File", "Take Photo (Camera)"])
    if input_method == "Take Photo (Camera)":
        uploaded_file = st.camera_input("Take a clear photo of the screen or film")
    else:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.file_uploader("Upload DICOM/Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Layout depends on mode
    if user_mode == "Healthcare Professional":
        col1, col2, col3 = st.columns([1, 1, 1.2])
        
        with col1:
            st.subheader("Original")
            image = Image.open(uploaded_file)
            st.image(image, width="stretch")
            
        with col2:
            st.subheader("AI Focus Area")
            heatmap_placeholder = st.empty()
            heatmap_placeholder.info("Run analysis to generate heatmap.")

        with col3:
            st.subheader("Clinical Insights")
            run_btn = st.button('Run Orchestrator', type="primary")
    else:
        # Patient Mode Layout (Simpler)
        st.image(uploaded_file, caption="Your Scan", width=300)
        run_btn = st.button('Check Health Status', type="primary")
        
    if run_btn:
        with st.spinner('Orchestrating Specialists...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image_data = Image.open(uploaded_file)
                # Convert to RGB if needed
                if image_data.mode != "RGB":
                    image_data = image_data.convert("RGB")
                image_data.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # CALL ROUTER
                # Map selectbox string to simple key
                part_key = "Chest"
                if "Brain" in body_part: part_key = "Brain"
                elif "Spine" in body_part: part_key = "Spine"
                
                # Determine method
                method = "unified" if "Unified" in engine_mode else "legacy"
                
                result = router.route_and_predict(tmp_path, body_part=part_key, method=method, metadata=meta_data)
                
                if "error" in result:
                    st.error(f"Router Error: {result['error']}")
                else:
                    preds = result["predictions"]
                    heatmap_data = result["heatmap"]
                    specialist_name = result["specialist"]
                    
                    if user_mode == "Healthcare Professional":
                        st.caption(f"Diagnosed by: {specialist_name}")
                
                    # ---- SPINE RESULTS ----
                    if part_key == "Spine":
                        st.markdown("### 🦴 Cervical Spine Fracture Analysis")
                        st.markdown("*Per-vertebra fracture probability — EfficientNetV2-B3 + BiGRU*")
                        st.markdown("---")
                        
                        SPINE_LABELS_ORDERED = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
                        CLASS_WEIGHTS = {'C1': 10.0, 'C2': 5.9, 'C3': 10.0, 
                                         'C4': 10.0, 'C5': 10.0, 'C6': 6.2, 'C7': 4.1}
                        
                        overall_score = preds.get('Patient_Overall', 0.0)
                        if overall_score > 0.6:
                            st.error(f"🚨 **Overall Patient Status: HIGH RISK** ({overall_score*100:.1f}%)")
                        elif overall_score > 0.35:
                            st.warning(f"⚠️ **Overall Patient Status: MODERATE RISK** ({overall_score*100:.1f}%)")
                        else:
                            st.success(f"✅ **Overall Patient Status: LOW RISK** ({overall_score*100:.1f}%)")
                        
                        st.markdown("#### Per-Vertebra Probability")
                        cols_spine = st.columns(7)
                        for i, label in enumerate(SPINE_LABELS_ORDERED):
                            score = preds.get(label, 0.0)
                            weight = CLASS_WEIGHTS.get(label, 1.0)
                            with cols_spine[i]:
                                color = "🔴" if score > 0.6 else ("🟡" if score > 0.35 else "🟢")
                                st.metric(label=f"{color} {label}", value=f"{score*100:.0f}%")
                                st.progress(float(score))
                                st.caption(f"w={weight}")
                        
                        st.markdown("---")
                        st.info("💡 High-probability vertebrae are flagged for urgent radiologist review. "
                                "Class weights reflect anatomical fracture rarity.")
                        
                        # Grad-CAM overlay if available
                        if np.max(heatmap_data) > 0 and user_mode == "Healthcare Professional":
                            overlay_img = overlay_heatmap(tmp_path, heatmap_data)
                            heatmap_placeholder.image(overlay_img, caption="Attention Map", use_container_width=True)

                    # ---- CHEST RESULTS ----
                    elif user_mode == "Healthcare Professional":
                        if np.max(heatmap_data) > 0:
                            overlay_img = overlay_heatmap(tmp_path, heatmap_data)
                            heatmap_placeholder.image(overlay_img, caption="Attention Map", use_container_width=True)
                        else:
                            heatmap_placeholder.info("Heatmap not available for this model.")

                        with col3:
                            display_comprehensive_findings(preds, mode="professional")

                    else:
                        # PATIENT MODE RESULT (Chest)
                        display_comprehensive_findings(preds, mode="patient")
                        st.info("We recommend sharing this result with a healthcare professional for a check-up.")
                        
                        # --- AI AGENT ANALYSIS ---
                        try:
                            agent_result = analyze_xray_with_agent(tmp_path)
                            
                            if agent_result.get("success"):
                                st.markdown("---")
                                provider = agent_result.get("provider", "AI")
                                st.markdown(f"### 🤖 AI Cross-Validation ({provider})")
                                
                                agent_data = agent_result.get("data", {})
                                findings = agent_data.get("findings", [])
                                
                                if findings:
                                    st.markdown("**AI Agent detected the following conditions:**")
                                    for f in findings:
                                        condition = f.get("condition", "Unknown")
                                        confidence = f.get("confidence", "Unknown")
                                        location = f.get("location", "")
                                        explanation = f.get("explanation", "")
                                        
                                        if confidence.lower() == "high":
                                            st.error(f"**{condition}** - {confidence} Confidence")
                                        elif confidence.lower() == "medium":
                                            st.warning(f"**{condition}** - {confidence} Confidence")
                                        else:
                                            st.info(f"**{condition}** - {confidence} Confidence")
                                        
                                        if location:
                                            st.caption(f"📍 Location: {location}")
                                        if explanation:
                                            st.caption(f"_{explanation}_")
                                    
                                    if agent_data.get("overall_impression"):
                                        st.markdown(f"**Overall:** {agent_data['overall_impression']}")
                                    
                                    if agent_data.get("recommendation"):
                                        st.success(f"**Recommendation:** {agent_data['recommendation']}")
                                elif "raw_analysis" in agent_data:
                                    st.markdown(agent_data["raw_analysis"])
                                else:
                                    st.success("AI Agent: No significant abnormalities detected.")
                        except Exception:
                            pass  # Silently skip AI Agent if unavailable

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

st.markdown("---")
st.caption("⚠️ **Disclaimer**: AI analysis is not a substitute for professional medical advice.")
