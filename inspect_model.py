import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = 'xray_model.h5'
try:
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
    
    print("\n--- Model Summary (Truncated) ---")
    # We want the last convolutional layer. 
    # Usually in DenseNet121 (common for X-Rays) it's 'conv5_block16_2_conv' or 'bn' or 'relu'.
    # We'll print the last 10 layers to identifying the candidate.
    for i, layer in enumerate(model.layers[-20:]):
        print(f"Index: {-20+i} | Name: {layer.name} | Type: {type(layer).__name__}")
        
except Exception as e:
    print(f"Error loading model: {e}")
