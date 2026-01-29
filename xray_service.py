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
            # Rebuild architecture manually to fix input shape/config mismatch
            base = EfficientNetB4(weights=None, include_top=False, input_shape=(380, 380, 3))
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.4)(x)
            x = Dense(128, activation='relu', name='dense_head')(x)
            outputs = Dense(1, activation='sigmoid', name='prediction_head')(x)
            self.model = Model(inputs=base.input, outputs=outputs)
            
            # Load weights by name to avoid layer count/topology strictness
            try:
                self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
                print("Weights loaded loosely (by_name=True). Verification recommended.")
            except Exception as e2:
                print(f"CRITICAL: Failed to load weights even by name: {e2}")

    def preprocess(self, img_path):
        # 380x380, Simple Rescale 1./255 (EfficientNet usually expects 0-255 but we follow training logic)
        img = image.load_img(img_path, target_size=(380, 380))
        x = image.img_to_array(img)
        x = x / 255.0 
        return np.expand_dims(x, axis=0)

    def format_results(self, probs):
        # Binary output: [score]
        score = float(probs[0]) if isinstance(probs, (np.ndarray, list)) else float(probs)
        return {"Pneumonia": score, "Normal": 1.0 - score}

        return {"Pneumonia": score, "Normal": 1.0 - score}

class PadChestPredictor(NIHPredictor):
    def __init__(self, model_path='padchest_sample_model.h5'):
        super().__init__(model_path)
        # Inherits everything from NIH for now, assuming compatible DenseNet structure
        pass

# --- ROUTER & ENSEMBLE ARCHITECTURE ---

class ChestEnsemble:
    """
    Combines predictions from multiple chest models to improve robustness.
    Weighted voting: NIH (70%) + PadChest (30%).
    """
    def __init__(self):
        print("Initializing Chest Ensemble...")
        # Load the specialists
        self.nih = NIHPredictor('xray_model.h5')
        self.pad = PadChestPredictor('padchest_sample_model.h5')
        
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
            
        # 2. Weighted Voting
        final_preds = {}
        
        # If both work, ensemble them
        if res_nih and res_pad:
            # We assume both models use the same NIH_LABELS set
            for label in NIH_LABELS:
                s1 = res_nih.get(label, 0.0)
                s2 = res_pad.get(label, 0.0)
                # Weighted Average: Trust NIH more (0.7) as it is the primary model
                final_preds[label] = (s1 * 0.7) + (s2 * 0.3)
                
            # Heatmap? We define the primary heatmap source as NIH
            heatmap = self.nih.make_gradcam(img_path)[0] # Just the map
            
        elif res_nih:
            final_preds = res_nih
            heatmap = self.nih.make_gradcam(img_path)[0]
        elif res_pad:
            final_preds = res_pad
            heatmap = self.pad.make_gradcam(img_path)[0]
        else:
            raise Exception("All ensemble models failed.")
            
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

