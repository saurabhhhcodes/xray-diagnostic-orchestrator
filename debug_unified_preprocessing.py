
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import os

# CONFIG
MODEL_PATH = 'Unified_Chest_SOTA_99_final.h5'
IMG_PATH = '/home/saurabh/.gemini/antigravity/brain/1970141d-9600-43e3-acb4-e9804e40d984/uploaded_media_0_1770191975229.png'

def load_and_preprocess(img_path, method='0-1'):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    
    if method == '0-1':
        x = x / 255.0
    elif method == '-1_1':
        x = (x / 127.5) - 1.0
    elif method == 'imagenet':
        x = x / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (x - mean) / std
    elif method == 'raw':
        pass # Keep 0-255

    # Dummy Metadata (Assuming High Risk)
    meta = np.array([0.65, 0.0]) # 65 yr old Male (Gender=0)
    
    return [np.expand_dims(x, axis=0), np.expand_dims(meta, axis=0)]

def run_debug():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print("Loading Model...")
    model = load_model(MODEL_PATH, compile=False)
    
    methods = ['0-1', '-1_1', 'imagenet', 'raw']
    print(f"\n--- Testing Preprocessing on {os.path.basename(IMG_PATH)} ---")
    
    for m in methods:
        inputs = load_and_preprocess(IMG_PATH, method=m)
        try:
            preds = model.predict(inputs, verbose=0)
            score = float(preds[0][0])
            print(f"Method [{m}]: Pneumonia Score = {score:.4f}")
            if score > 0.5:
                print(f"  >>> MATCH! [{m}] yields POSITIVE detection.")
        except Exception as e:
            print(f"Method [{m}] Failed: {e}")

if __name__ == "__main__":
    run_debug()
