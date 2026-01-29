
import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Imports
c1 = """# Cell 1: Imports & System Setup
import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision

# Mixed Precision for T4 Speedup
mixed_precision.set_global_policy('mixed_float16')

SEED = 2026
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"TensorFlow {tf.__version__}")"""

# Cell 2: Data Loading
c2 = """# Cell 2: RSNA Data Loading
print("Searching for RSNA data...")
# Look for stage_2_train_labels.csv or similar
csv_candidates = glob('/kaggle/input/**/*.csv', recursive=True)
target_csv = [c for c in csv_candidates if 'stage_2_train_labels' in c or 'train' in c.lower()][0]
print(f"Using CSV: {target_csv}")

df = pd.read_csv(target_csv)
# RSNA specific: Clean duplicates (PatientId)
df = df.drop_duplicates(subset=['patientId'])

# Paths
img_dir = os.path.dirname(glob('/kaggle/input/**/*.png', recursive=True)[0])
df['path'] = df['patientId'].apply(lambda x: os.path.join(img_dir, x + '.png'))

# Target
df['class'] = df['Target'].apply(lambda x: 'Pneumonia' if x == 1 else 'Normal')

print(df['class'].value_counts())

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=SEED)"""

# Cell 3: Augmentation
c3 = """# Cell 3: Generators (EfficientNetB4 380x380)
IMG_SIZE = (380, 380)
BATCH_SIZE = 16 

train_idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_idg = ImageDataGenerator(rescale=1./255)

train_gen = train_idg.flow_from_dataframe(train_df, x_col='path', y_col='class', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
val_gen = val_idg.flow_from_dataframe(val_df, x_col='path', y_col='class', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')"""

# Cell 4: Model
c4 = """# Cell 4: EfficientNetB4
base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3))
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])"""

# Cell 5: Train
c5 = """# Cell 5: Training Loop
checkpoint = ModelCheckpoint('rsna_pneumonia.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_auc', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.2, patience=2, min_lr=1e-6)

history = model.fit(train_gen, epochs=25, validation_data=val_gen, callbacks=[checkpoint, early_stop, reduce_lr])"""

nb['cells'] = [nbf.v4.new_code_cell(c) for c in [c1, c2, c3, c4, c5]]

with open('rsna_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("RSNA Notebook generated.")
