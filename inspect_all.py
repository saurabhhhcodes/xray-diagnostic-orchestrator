
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

models = ['xray_model.h5', 'rsna_pneumonia.h5', 'padchest_sample_model.h5']

for m_path in models:
    if os.path.exists(m_path):
        print(f"\n--- INSPECTING: {m_path} ---")
        try:
            model = load_model(m_path, compile=False)
            print(f"Input Shape: {model.input_shape}")
            
            # Print last few layers to find the conv layer for GradCAM and output details
            print("Last 10 Layers:")
            for i, layer in enumerate(model.layers[-10:]):
                print(f"  {i}: {layer.name} ({type(layer).__name__}) -> Output: {layer.output_shape}")
                
        except Exception as e:
            print(f"FAILED to load {m_path}: {e}")
    else:
        print(f"File not found: {m_path}")
