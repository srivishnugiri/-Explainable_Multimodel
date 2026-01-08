# ================= IMPORTS =================
import os, librosa, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf  # ADD THIS LINE

# ================= CONFIG =================
np.random.seed(42)
tf.random.set_seed(42)
classes = ["Alzheimer", "Healthy"]
DATASET_PATH = "E:\\Project\\Dataset\\Audio"
MODEL_PATH = "Alzheimer_Audio_ConvLSTM_Fixed.h5"
EPOCHS, BATCH_SIZE = 50, 8  # Smaller batch size for better generalization

# ================= ENHANCED FEATURE EXTRACTION =================
def extract_enhanced_features(file_path, max_pad_len=100):
    try:
        signal, sr = librosa.load(file_path, sr=22050)
        signal, _ = librosa.effects.trim(signal, top_db=20)
        
        # Multiple feature types
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=2048, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=2048, hop_length=512)
        tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
        
        # Stack features
        features = np.vstack([mfcc, chroma, spectral_contrast, tonnetz])
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Padding/truncation
        if features.shape[1] < max_pad_len:
            pad = max_pad_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad)), mode='constant')
        else:
            features = features[:, :max_pad_len]
            
        return features.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ================= LOAD DATA WITH AUGMENTATION =================
def load_dataset_balanced(path):
    X, y = [], []
    for cls in classes:
        folder = os.path.join(path, cls)
        if not os.path.exists(folder):
            print(f"Warning: {folder} does not exist")
            continue
            
        for f in os.listdir(folder):
            if f.endswith(".wav"):
                feat = extract_enhanced_features(os.path.join(folder, f))
                if feat is not None:
                    X.append(feat)
                    y.append(cls)
    
    if len(X) == 0:
        raise ValueError("No features extracted! Check dataset path and file format.")
        
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_dataset_balanced(DATASET_PATH)

# Analyze class distribution
print(f"Class distribution: {np.unique(y, return_counts=True)}")

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)
y_cat = to_categorical(y_enc)

# Calculate class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_enc), y=y_enc)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights: {class_weight_dict}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ================= IMPROVED MODEL WITH REGULARIZATION =================
def build_improved_model(input_shape, num_classes):
    model = Sequential([
        # First Conv Block
        Conv1D(32, 5, activation='relu', padding='same', input_shape=input_shape,
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),
        Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model

# ================= ENHANCED TRAINING =================
if not os.path.exists(MODEL_PATH):
    model = build_improved_model((X_train.shape[1], X_train.shape[2]), len(classes))
    model.summary()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7
        )
    ]
    
    print("Training improved model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # Handle class imbalance
        verbose=1,
        shuffle=True
    )
    
    model.save(MODEL_PATH)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
else:
    model = load_model(MODEL_PATH)
    print("Loaded pre-trained model")
# ================= COMPREHENSIVE EVALUATION =================
print("\n" + "="*50)
print("COMPREHENSIVE EVALUATION")
print("="*50)

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=classes, yticklabels=classes,
            cbar_kws={'label': 'Count'})
plt.title("Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Confidence Analysis
confidences = np.max(y_pred_proba, axis=1)
correct_predictions = y_pred == y_true

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist([confidences[correct_predictions], confidences[~correct_predictions]], 
         bins=10, label=['Correct', 'Incorrect'], 
         color=['green', 'red'], alpha=0.7, stacked=True)
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.legend()
plt.title("Prediction Confidence Distribution")

plt.subplot(1, 2, 2)
# Class-wise confidence
for i, class_name in enumerate(classes):
    class_mask = y_true == i
    if np.any(class_mask):
        plt.hist(confidences[class_mask], alpha=0.6, label=f'{class_name}', bins=8)
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.legend()
plt.title("Class-wise Confidence Distribution")

plt.tight_layout()
plt.show()

# Overfitting Analysis - FIXED THIS PART
train_metrics = model.evaluate(X_train, y_train, verbose=0)
test_metrics = model.evaluate(X_test, y_test, verbose=0)

train_accuracy = train_metrics[1]  # Second metric is accuracy
test_accuracy = test_metrics[1]
train_loss = train_metrics[0]      # First metric is loss
test_loss = test_metrics[0]

print(f"\n🔍 OVERFITTING ANALYSIS:")
print(f"   Training Accuracy: {train_accuracy:.4f}")
print(f"   Test Accuracy:     {test_accuracy:.4f}")
print(f"   Difference:        {train_accuracy - test_accuracy:+.4f}")
print(f"   Training Loss:     {train_loss:.4f}")
print(f"   Test Loss:         {test_loss:.4f}")
print(f"   Difference:        {train_loss - test_loss:+.4f}")

if (train_accuracy - test_accuracy) > 0.1:
    print("   ⚠️  WARNING: Significant overfitting detected!")
elif (train_accuracy - test_accuracy) > 0.05:
    print("   ℹ️  Moderate overfitting")
else:
    print("   ✅ Good generalization - minimal overfitting")

print(f"\n🎯 Final Test Accuracy: {test_accuracy:.3f}")
print("✅ Enhanced audio classification complete.")