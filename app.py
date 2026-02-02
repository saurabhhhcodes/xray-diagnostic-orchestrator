import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import matplotlib.cm as cm
from xray_service import NIHPredictor, RSNAPredictor, PadChestPredictor

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

from xray_service import DiagnosticRouter

st.title("🩻 Sahayak Diagnostic Orchestrator")

# --- USER MODE SELECTION ---
user_mode = st.sidebar.radio("User Mode", ["Patient", "Healthcare Professional"])

# --- ROUTER CONFIGURATION ---
st.sidebar.title("⚙️ Analysis Settings")
body_part = st.sidebar.selectbox(
    "Select Body Part / Scan Type",
    ["Chest X-Ray", "Brain MRI (Coming Soon)", "Bone Fracture (Coming Soon)", "Retinal Scan (Coming Soon)"]
)

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
                elif "Bone" in body_part: part_key = "Bone"
                
                result = router.route_and_predict(tmp_path, body_part=part_key)
                
                if "error" in result:
                    st.error(f"Router Error: {result['error']}")
                else:
                    preds = result["predictions"]
                    heatmap_data = result["heatmap"]
                    specialist_name = result["specialist"]
                    
                    if user_mode == "Healthcare Professional":
                        st.caption(f"Diagnosed by: {specialist_name}")
                
                    # RENDER RESULTS
                    if user_mode == "Healthcare Professional":
                        if np.max(heatmap_data) > 0:
                            overlay_img = overlay_heatmap(tmp_path, heatmap_data)
                            heatmap_placeholder.image(overlay_img, caption=f"Attention Map", width="stretch")
                        else:
                            heatmap_placeholder.info("Heatmap not available for this model.")

                        user_col = col3
                        
                        sort_p = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True))
                        top_l = list(sort_p.keys())[0]
                        top_s = sort_p[top_l]
                        
                        with user_col:
                            if top_l == "Normal" or (top_l == "No Finding" and top_s > 0.5):
                                    st.success(f"**LIKELY NORMAL**\n\nConfidence: {top_s*100:.1f}%")
                            else:
                                    st.error(f"**FINDING: {top_l.upper()}**\n\nConfidence: {top_s*100:.1f}%")
                            st.markdown("---")
                            
                            c = 0
                            for l, s in sort_p.items():
                                if c >= 4: break
                                with st.expander(f"{l}: {s*100:.1f}%"):
                                    st.progress(s)
                                    st.caption(DISEASE_DETAILS.get(l, ""))
                                c+=1

                    else:
                        # PATIENT MODE RESULT
                        sorted_preds = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True))
                        top_label = list(sorted_preds.keys())[0]
                        top_score = sorted_preds[top_label]
                        
                        if top_label == "Normal" or (top_label == "No Finding" and top_score > 0.5):
                            st.canvas = st.balloons()
                            st.success("### ✅ Likely Healthy")
                            st.write(PATIENT_FRIENDLY_DESCRIPTIONS.get("Normal"))
                        else:
                            st.warning(f"### ⚠️ Attention Needed: {top_label}")
                            st.write(PATIENT_FRIENDLY_DESCRIPTIONS.get(top_label, "Please consult a doctor."))
                            st.progress(top_score)
                            st.info("ℹ️ **Methodology**: Scores are fused from NIH, PadChest, and RSNA Specialists.")

                            # --- WEB VERIFICATION ---
                            st.markdown("### 🌐 External Verification")
                            search_query = f"{top_label} X-Ray radiology radiopedia"
                            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
                            
                            st.markdown(f"""
                                <a href="{search_url}" target="_blank">
                                    <button style="
                                        background-color: #4CAF50;
                                        color: white;
                                        padding: 10px 24px;
                                        border: none;
                                        border-radius: 4px;
                                        cursor: pointer;
                                        font-size: 16px;
                                        width: 100%;">
                                        🔍 Verify '{top_label}' on Web
                                    </button>
                                </a>
                                <p style="font-size:0.8em; color:gray; text-align:center; margin-top:5px;">
                                    Opens a medical search for reference images.
                                </p>
                            """, unsafe_allow_html=True)
                            st.info("We recommend sharing this result with a healthcare professional for a check-up.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

st.markdown("---")
st.caption("⚠️ **Disclaimer**: AI analysis is not a substitute for professional medical advice.")
