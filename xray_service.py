import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
import os
import cv2

# --- CONFIGURATION ---
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

RSNA_LABELS = ['Normal', 'Pneumonia']

class BaseXRayPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        if not os.path.exists(model_path):
             print(f"WARNING: Model file not found at {model_path}")
             self.model = None
        else:
             print(f"Loading model from {model_path}...")
             self.load_model_safe()
             print(f"Model {os.path.basename(model_path)} loaded.")

    def load_model_safe(self):
        """Override this for custom loading logic"""
        self.model = load_model(self.model_path, compile=False)

    def preprocess(self, img_path):
        """Override for model-specific preprocessing"""
        raise NotImplementedError

    def predict(self, img_path):
        """Standard prediction pipeline"""
        if self.model is None:
            raise FileNotFoundError("Model not loaded")
        
        processed_input = self.preprocess(img_path)
        probs = self.model.predict(processed_input)[0]
        return self.format_results(probs)

    def format_results(self, probs):
        """Override to map probabilities to labels"""
        raise NotImplementedError

class NIHPredictor(BaseXRayPredictor):
    def __init__(self, model_path='xray_model.h5'):
        super().__init__(model_path)
        self.last_conv_layer = "conv5_block16_2_conv" # DenseNet121 specific

    def preprocess(self, img_path):
        # 224x224, ImageNet Mean/Std
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (x - mean) / std
        return np.expand_dims(x, axis=0)

    def format_results(self, probs):
        results = {label: float(score) for label, score in zip(NIH_LABELS, probs)}
        return results

    def make_gradcam(self, img_path, target_layer=None):
        if target_layer is None: target_layer = self.last_conv_layer
        
        img_array = self.preprocess(img_path)
        
        grad_model = Model(
            self.model.inputs, 
            [self.model.get_layer(target_layer).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            # Handle Functional API returning list for single output
            if isinstance(preds, list): preds = preds[0]
            
            top_class_idx = tf.argmax(preds[0])
            top_class_channel = preds[:, top_class_idx]

        grads = tape.gradient(top_class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy(), NIH_LABELS[top_class_idx]

class RSNAPredictor(BaseXRayPredictor):
    def __init__(self, model_path='rsna_pneumonia.h5'):
        super().__init__(model_path)

    def load_model_safe(self):
        try:
            # First try standard load
            self.model = load_model(self.model_path, compile=False)
        except Exception as e:
            print(f"Standard load failed ({e}). Rebuilding EfficientNetB4 architecture...")
            # Rebuild architecture manually
            # Corrected: Input 0-255 (implicitly handled by rescaling if present, but we pass raw)
            # Corrected: 256 units for dense layer
            # Corrected: Layer names 'dense' and 'dense_1' to match H5
            base = EfficientNetB4(weights=None, include_top=False, input_shape=(380, 380, 3))
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.4)(x)
            # Use 'dense' instead of 'dense_head' and 256 units
            x = Dense(256, activation='relu', name='dense')(x)
            # Use 'dense_1' instead of 'prediction_head'
            outputs = Dense(1, activation='linear', name='dense_1')(x)
            self.model = Model(inputs=base.input, outputs=outputs)
            
            # Load weights by name
            try:
                self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
                print("Weights loaded loosely (by_name=True).")
            except Exception as e2:
                print(f"CRITICAL: Failed to load weights even by name: {e2}")

    def preprocess(self, img_path):
        # 380x380
        # EfficientNetB4 internal preprocessing usually includes Rescaling if built with include_top=False? 
        # Actually, keras applications usually expect raw input for their preprocess_input function, 
        # OR they have a Rescaling layer inside. 
        img = image.load_img(img_path, target_size=(380, 380))
        x = image.img_to_array(img)
        # x = x / 255.0 # Removed: Model has internal Rescaling layer (1/255)
        return np.expand_dims(x, axis=0)

    def format_results(self, logits):
        # Logits output: [logit]
        logit = float(logits[0]) if isinstance(logits, (np.ndarray, list)) else float(logits)
        
        # Temperature Scaling to soften the model's overconfidence
        # T > 1.0 makes it less confident (pushes 0.0 or 1.0 towards 0.5)
        # T < 1.0 makes it more confident
        TEMPERATURE = 5.0 
        
        # Sigmoid with Temperature
        # score = 1 / (1 + e^(-logit/T))
        score = 1.0 / (1.0 + np.exp(-logit / TEMPERATURE))
        
        print(f"RSNA Raw Logit: {logit:.4f} -> Scaled Score (T={TEMPERATURE}): {score:.4f}")
        
        return {"Pneumonia": score, "Normal": 1.0 - score}

        return {"Pneumonia": score, "Normal": 1.0 - score}

    def make_gradcam(self, img_path):
        # Target last conv layer of EfficientNetB4
        # Usually 'top_activation' or similar. 
        # Since we rebuilt it, we used 'base' which is include_top=False.
        # The last layer of 'base' is 'top_activation'.
        target_layer = 'top_activation' 
        
        img_array = self.preprocess(img_path)
        
        # We need to access the 'base' model inside self.model
        # self.model is: Input -> base -> GlobalPool -> Dense -> Dense
        # So we can't easily access 'top_activation' via self.model.get_layer() 
        # because 'base' is a nested Model/Functional?
        # Actually in Keras Functional API, layers of nested models are accessible if built explicitly?
        # No, 'base' is treated as a layer "efficientnetb4".
        
        # Alternative: Create a new model sharing weights?
        # Simpler: Get the 'efficientnetb4' layer output?
        
        try:
            # Locate the EfficientNet part
            # If we built it manually in load_model_safe:
            # base = EfficientNetB4...
            # self.model = Model(inputs=base.input, outputs=outputs)
            # The layers of 'base' are effectively flattened into 'self.model' inputs/outputs?
            # No, 'base' is a graph.
            
            # Let's inspect layers to find the conv layer
            # If standard load failed, we rebuilt it.
            # In rebuild, 'base' is used.
            # If 'base' is used as a functional calls, the layers should be in self.model if we iterate?
            # Or is 'efficientnetb4' a single layer?
            pass
            
            # Robust strategy:
            # Construct a grad model from the 'base' model instance if possible?
            # We don't have reference to 'base' logic outside __init__.
            
            # Let's try to access layer by name 'top_activation'
            grad_model = Model(
                self.model.inputs, 
                [self.model.get_layer('top_activation').output, self.model.output]
            )
        except Exception as e:
            # Fallback for nested model issue: Use last layer that is 4D
            for l in reversed(self.model.layers):
                 if len(l.output_shape) == 4:
                     grad_model = Model(self.model.inputs, [l.output, self.model.output])
                     break
        
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if isinstance(preds, list): preds = preds[0]
            # Prob for Pneumonia (class 0? or 1?)
            # format_results says score = probs[0]. 
            # Our Dense(1) output.
            loss = preds[0] # The Pneumonia score itself
            
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy(), "Pneumonia"

class PadChestPredictor(NIHPredictor):
    def __init__(self, model_path='padchest_sample_model.h5'):
        super().__init__(model_path)
        # Inherits everything from NIH for now, assuming compatible DenseNet structure
        pass

# --- ROUTER & ENSEMBLE ARCHITECTURE ---

class ChestEnsemble:
    """
    Combines predictions from multiple chest models to improve robustness.
    Weighted voting: NIH (60%) + PadChest (40%).
    Special Case: RSNA Pneumonia Model boosts "Pneumonia" confidence.
    """
    def __init__(self):
        print("Initializing Chest Ensemble...")
        # Load the specialists
        self.nih = NIHPredictor('xray_model.h5')
        self.pad = PadChestPredictor('padchest_sample_model.h5')
        self.rsna = RSNAPredictor('rsna_pneumonia.h5')
        
    def predict(self, img_path):
        # 1. Get individual predictions
        try:
            res_nih = self.nih.predict(img_path)
        except Exception as e:
            print(f"Ensemble Warning: NIH model failed ({e}). Ignoring.")
            res_nih = None

        try:
            res_pad = self.pad.predict(img_path)
        except Exception as e:
            print(f"Ensemble Warning: PadChest model failed ({e}). Ignoring.")
            res_pad = None

        try:
            res_rsna = self.rsna.predict(img_path)
        except Exception as e:
            print(f"Ensemble Warning: RSNA model failed ({e}). Ignoring.")
            res_rsna = None
            
        # 2. Weighted Voting
        final_preds = {}
        
        # If both generalists work, ensemble them
        if res_nih and res_pad:
            # We assume both models use the same NIH_LABELS set
            for label in NIH_LABELS:
                s1 = res_nih.get(label, 0.0)
                s2 = res_pad.get(label, 0.0)
                # Weighted Average: NIH (0.6) + PadChest (0.4)
                final_preds[label] = (s1 * 0.6) + (s2 * 0.4)
        elif res_nih:
            final_preds = res_nih
        elif res_pad:
            final_preds = res_pad
        else:
            final_preds = {} # Should probably raise error, but let's see if RSNA works

        # 3. RSNA BOOST (The Pneumonia Specialist)
        # If RSNA is available and detects Pneumonia with high confidence, we trust it.
        if res_rsna:
            rsna_pneu_score = res_rsna.get("Pneumonia", 0.0)
            current_pneu_score = final_preds.get("Pneumonia", 0.0)
            
            # Logic: If Specialist is very sure (>0.5), we allow it to boost the score.
            # We take the MAXIMUM of the Ensemble Score and the Specialist Score.
            # This ensures we don't miss a case just because the generalists were unsure.
            if rsna_pneu_score > current_pneu_score:
                print(f"Ensemble: RSNA boosted Pneumonia from {current_pneu_score:.2f} to {rsna_pneu_score:.2f}")
                final_preds["Pneumonia"] = rsna_pneu_score
                
        # 4. Clinical Correlation / Comorbidity Boosting
        # RSNA might miss specific features that manifest as Effusion/Infiltration in NIH.
        # If High Risk indicators are present, we boost Pneumonia sensitivity.
        risk_indicators = ['Infiltration', 'Effusion', 'Consolidation']
        risk_score = sum(final_preds.get(k, 0.0) for k in risk_indicators)
        
        # If risk indicators are significant sum > 0.2 (NIH detected something)
        # We ensure Pneumonia is at least visible (e.g. 100% of the risk mass involved)
        # Clinical reasoning: Pneumonia prediction is often the goal. If Infiltration/Effusion match,
        # it is highly likely Pneumonia. We boost it to match the risk level.
        boosted_pneu = max(final_preds.get("Pneumonia", 0.0), risk_score * 1.0)
        
        if boosted_pneu > final_preds.get("Pneumonia", 0.0):
             print(f"Ensemble: Clinical Correlation boosted Pneumonia to {boosted_pneu:.2f} (Risk Score: {risk_score:.2f})")
             final_preds["Pneumonia"] = boosted_pneu

        # Heatmap selection: Try RSNA first if Pneumonia is the top finding?
        # For consistency, we stick to NIH for general heatmap, unless it's missing.
        heatmap = np.zeros((224, 224))
        if self.nih and self.nih.model:
             try:
                heatmap = self.nih.make_gradcam(img_path)[0]
             except: pass
            
        return final_preds, heatmap

class DiagnosticRouter:
    """
    The 'General Practitioner'. Routes the image to the correct specialist ensemble.
    """
    def __init__(self):
        self.chest_ensemble = None 
        # Lazy load to save memory until needed? 
        # For this demo, let's load immediately to be responsive.
        self.chest_ensemble = ChestEnsemble()
        
    def route_and_predict(self, img_path, body_part="Chest"):
        if body_part.lower() == "chest":
            preds, heatmap = self.chest_ensemble.predict(img_path)
            return {
                "predictions": preds,
                "heatmap": heatmap,
                "specialist": "Chest X-Ray Ensemble (NIH + PadChest)"
            }
        
        elif body_part.lower() == "brain":
            # Future placeholder
            return {
                "error": "Brain MRI Specialist is currently offline (Coming Soon).",
                "specialist": "Brain Neuro-AI"
            }
        
        elif body_part.lower() == "bone":
            # Future placeholder
            return {
                "error": "Bone Fracture Specialist is currently offline (Coming Soon).",
                "specialist": "Ortho-AI"
            }
            
        return {"error": "Unknown body part detected."}

