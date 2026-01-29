import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Imports & Setup
text_1 = """# Cell 1: Imports & System Setup
import numpy as np
import pandas as pd
import os
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Reproducibility
SEED = 2026
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow Version: {tf.__version__}")"""

# Cell 2: Smart Loader
text_2 = """# Cell 2: Smart Data Loader (Robust)
# --- SMART PATH DETECTION ---
print("Searching for dataset files...")
csv_files = glob('/kaggle/input/**/Data_Entry_2017.csv', recursive=True)

if not csv_files:
    raise FileNotFoundError("Could not find Data_Entry_2017.csv. Please check the 'Input' sidebar in Kaggle and add the 'NIH Chest X-rays' dataset.")

csv_path = csv_files[0]
dataset_root = os.path.dirname(csv_path)
print(f"Found dataset at: {dataset_root}")

# Map all image paths
image_paths = glob(os.path.join(dataset_root, '**', '*.png'), recursive=True)
print(f"Found {len(image_paths)} images.")

if len(image_paths) == 0:
    raise FileNotFoundError("Found CSV but no images! Check dataset structure.")

path_map = {os.path.basename(x): x for x in image_paths}

# --- DATA PROCESSING ---
data = pd.read_csv(csv_path)
data['path'] = data['Image Index'].map(path_map.get)
data = data[data['path'].notnull()]

# Handle Multi-Labels
all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print(f"Classes: {all_labels}")

for label in all_labels:
    data[label] = data['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)

# Split by Patient ID (Prevent Data Leakage)
train_ids, val_ids = train_test_split(data['Patient ID'].unique(), test_size=0.2, random_state=SEED)
train_df = data[data['Patient ID'].isin(train_ids)]
val_df = data[data['Patient ID'].isin(val_ids)]

print(f"Final Train Size: {len(train_df)} | Validation Size: {len(val_df)}")"""

# Cell 3: Augmentation
text_3 = """# Cell 3: Data Augmentation & Generators
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32

core_idg = ImageDataGenerator(
    rescale=1./255, 
    samplewise_center=True, 
    samplewise_std_normalization=True, 
    horizontal_flip=True, 
    vertical_flip=False, 
    rotation_range=20, 
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

def get_generator(df):
    return core_idg.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col='path',
        y_col=all_labels,
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=IMG_SIZE
    )

train_gen = get_generator(train_df)
val_gen = get_generator(val_df)"""

# Cell 4: Model
text_4 = """# Cell 4: Build Model (DenseNet121)
# Transfer Learning
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(all_labels), activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['binary_accuracy', tf.keras.metrics.AUC(multi_label=True, name='auc')])"""

# Cell 5: Train
text_5 = """# Cell 5: Train & Save
# Fast Training: Check 10% of validation set per epoch to save time
val_steps = len(val_gen) // 10 
train_steps = len(train_gen)

checkpoint = ModelCheckpoint(
    'xray_model.h5', 
    monitor='val_auc', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

early_stop = EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=1)

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps, 
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=10, 
    callbacks=[checkpoint, early_stop, reduce_lr]
)"""

nb['cells'] = [
    nbf.v4.new_code_cell(text_1),
    nbf.v4.new_code_cell(text_2),
    nbf.v4.new_code_cell(text_3),
    nbf.v4.new_code_cell(text_4),
    nbf.v4.new_code_cell(text_5)
]

with open('kaggle_xray_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
