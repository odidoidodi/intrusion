import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 1. SMART PATH RESOLUTION
# ============================================
print("=" * 60)
print("NETWORK INTRUSION DETECTION - CICIDS2017")
print("=" * 60)
print("\n[1] LOCATING CSV FILES...")

current_dir_files = glob.glob("*WorkingHours*.csv")

if len(current_dir_files) >= 4:
    DATA_FOLDER = "."
    print(f"Detected CSV files in current directory.")
else:
    if os.path.exists("datasets"):
        DATA_FOLDER = "datasets"
    elif os.path.exists("IP_attack/datasets"):
        DATA_FOLDER = "IP_attack/datasets"
    else:
        raise FileNotFoundError("Cannot find CSV files.")

csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError("No .csv files found!")

print(f"Found {len(csv_files)} CSV files to process:\n")

# ============================================
# 2.DATA LOADER
# ============================================
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
    
    df.columns = df.columns.str.strip()
    df = df.drop(columns=df.columns[df.columns == ''], errors='ignore')
    df = df.loc[:, df.columns.notna()]
    df = df.loc[:, ~df.columns.duplicated()] 
    
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
            
    if label_col is None:
        print(f" 'Label' missing in {os.path.basename(file_path)}. Skipping.")
        return None
        
    first_col = df.columns[0]
    try:
        if df[first_col].equals(pd.Series(range(len(df)))):
            df = df.drop(columns=[first_col])
    except:
        pass
        
    return df


# ============================================
# 3. LOAD ALL FILES & MEMORY OPTIMIZATION (STABLE)
# ============================================
df_list = []
for file in csv_files:
    file_name = os.path.basename(file)
    df_temp = load_and_clean_data(file_path=file)
    
    if df_temp is not None:
        # Clean labels
        df_temp['Label'] = df_temp['Label'].astype(str).str.strip()
        df_temp = df_temp[~df_temp['Label'].isin(['nan', 'None', '', 'Label'])]
        
        # --- CRITICAL FIX: Pre-sample huge files ---
        # If a single file is massive (e.g., > 1.5M rows), reduce it immediately 
        # to prevent the total list from exceeding RAM during concatenation.
        if len(df_temp) > 1500000: 
            print(f"  -> File {file_name} is large ({len(df_temp):,}). Sampling down to 500k...")
            df_temp = df_temp.sample(n=500000, random_state=42).reset_index(drop=True)
        
        df_list.append(df_temp)
        print(f"{file_name:<55} | Rows: {len(df_temp):>8,}")

print("\nConcatenating all datasets...")
# Concatenate the (now smaller) list of dataframes. 
# Since we pre-sampled huge files, this won't spike memory as hard.
df = pd.concat(df_list, ignore_index=True)

# Free memory from the list immediately after concat finishes
del df_list 
import gc; gc.collect() 

print(f"Total Combined Dataset Shape: {df.shape}")

# Final sampling to get exactly what we need for training (e.g., 2M rows)
MAX_ROWS = 2000000 
if len(df) > MAX_ROWS:
    print(f"Dataset is large ({len(df):,} rows). Sampling final set of {MAX_ROWS:,} rows...")
    
    # Use numpy for efficient sampling
    indices = np.random.choice(len(df), size=MAX_ROWS, replace=False)
    df_sampled = df.iloc[indices].reset_index(drop=True)
    
    del df 
    gc.collect()
    df = df_sampled


# ============================================
# 4. PREPROCESSING (FIXED)
# ============================================
print("\n[2] PREPROCESSING DATA...")

df = df.dropna(subset=['Label'])
df['Label'] = df['Label'].astype(str).str.strip()
df = df[~df['Label'].str.lower().isin(['nan', 'none', '', 'label'])]

X = df.drop('Label', axis=1)
y = df['Label']

print(f"Label Distribution:\n{y.value_counts()}\n")

# --- ROBUST CLEANING SEQUENCE ---

# 1. Convert ALL columns to numeric first (Strings become NaN)
X = X.apply(pd.to_numeric, errors='coerce')

# 2. Replace Infinity with NaN (StandardScaler hates inf)
# This MUST happen after conversion to catch any "inf" strings that became floats
X = X.replace([np.inf, -np.inf], np.nan)

# 3. Drop columns that are entirely empty after conversion
X = X.dropna(axis=1, how='all')

# 4. Fill remaining NaNs with median (robust to outliers)
X = X.fillna(X.median())

# 5. Final check: fill any remaining NaNs with 0
X = X.fillna(0)

feature_names = X.columns.tolist()
print(f"Final Features Count: {X.shape[1]}")

# ============================================
# 5. ENCODING & SCALING
# ============================================
print("\n[3] ENCODING & SCALING...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
class_names = list(label_encoder.classes_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32) 

# ============================================
# 6. TRAIN-TEST SPLIT (With Safety Check for Rare Classes)
# ============================================
print("\n[4] SPLITTING DATA...")

try:
    # First Split: Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.10, random_state=42, stratify=y_encoded
    )
    
    # Second Split: Train/Val
    # Note: If a class has very few samples (e.g., Heartbleed with 2), 
    # the second split might fail if only 1 sample remains in X_train.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42, stratify=y_train
    )

except ValueError as e:
    print(f"Stratification error (likely due to rare classes): {e}")
    print("Retrying second split without strict stratification...")
    
    # Fallback for the second split if rare classes cause issues
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=42
    )

print(f"Training: {X_train.shape[0]:,} | Validation: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ============================================
# 7. BUILD NEURAL NETWORK
# ============================================
print("\n[5] BUILDING NEURAL NETWORK...")
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.InputLayer(input_shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# 8. TRAIN MODEL
# ============================================
print("\n[6] TRAINING MODEL...")

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=2048,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
# ============================================
# 9. ADVANCED EVALUATION & METRICS (FIXED)
# ============================================
print("\n" + "=" * 60)
print("[7] EVALUATION METRICS")
print("=" * 60)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

# FIX: Generate report with explicit labels to handle missing rare classes in test set
report_dict = classification_report(
    y_test, 
    y_pred, 
    target_names=class_names, 
    zero_division=0, 
    output_dict=True,
    labels=range(num_classes) # <--- CRITICAL FIX: Forces all 29 classes to be included
)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0, labels=range(num_classes)))

# Matthews Correlation Coefficient (Best metric for imbalanced data)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"--- Additional Metrics ---")
print(f"Test Accuracy: {acc:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# ============================================
# 10. PROFESSIONAL PLOTS (FIXED)
# ============================================
print("\n[8] GENERATING PLOTS...")

fig = plt.figure(figsize=(20, 12))

# --- Plot 1: Training History ---
ax1 = plt.subplot(2, 2, 1)
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
ax1.set_title('Model Accuracy over Epochs', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# --- Plot 2: Loss History ---
ax2 = plt.subplot(2, 2, 2)
ax2.plot(history.history['loss'], label='Train Loss', color='blue')
ax2.plot(history.history['val_loss'], label='Val Loss', color='orange')
ax2.set_title('Model Loss over Epochs', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# --- Plot 3: Confusion Matrix ---
ax3 = plt.subplot(2, 2, 3)
# FIX: Ensure matrix is 29x29 even if some classes are missing in test set
cm = confusion_matrix(y_test, y_pred, labels=range(num_classes)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=class_names if num_classes <= 10 else False,
            yticklabels=class_names if num_classes <= 10 else False)
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax3.get_yticklabels(), rotation=0)

# --- Plot 4: Precision, Recall, F1-Score per Class ---
ax4 = plt.subplot(2, 2, 4)
# Use the report_dict we already generated above (avoids re-running and crashing again)
metrics_df = pd.DataFrame(report_dict).T
# Remove accuracy and averages rows so we only plot individual classes
metrics_df = metrics_df[~metrics_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]

metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax4, width=0.8, edgecolor='black')
ax4.set_title('Precision, Recall, and F1-Score by Attack Type', fontsize=14, fontweight='bold')
ax4.set_ylabel('Score (0.0 to 1.0)')
ax4.set_xlabel('Attack Class')
ax4.set_ylim(0, 1.1)
ax4.legend(loc='lower right')
ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('complete_evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 11. SAVE ARTIFACTS
# ============================================
print("\n[9] SAVING MODEL AND ARTIFACTS...")
model.save('cicids_intrusion_model.keras')
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("\n" + "=" * 60)
print("TRAINING & EVALUATION PIPELINE COMPLETE!")
print("=" * 60)