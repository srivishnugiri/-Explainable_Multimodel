# ================= IMPORTS =================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers, callbacks

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5  # AD, CN, EMCI, LMCI, MCI

# ================= LOCAL PATHS =================
# Change these paths to your local dataset folders
base_dir = r"E:\data\ADNI_Data"   # <-- Main dataset folder
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Folder to save trained models locally
save_dir = os.path.join(base_dir, "models")
os.makedirs(save_dir, exist_ok=True)

# ================= DATA GENERATORS WITH CLASS WEIGHTS =================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ================= HANDLE CLASS IMBALANCE =================
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ================= MODEL ARCHITECTURE =================
def create_improved_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze initially

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )

    return model, base_model

model, base_model = create_improved_model()
model.summary()

# ================= CALLBACKS =================
best_model_path = os.path.join(save_dir, "vgg16_adni_best.keras")
checkpoint_cb = callbacks.ModelCheckpoint(
    best_model_path,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

early_stopping_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# ================= TRAIN MODEL =================
print("🚀 Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_cb],
    class_weight=class_weight_dict,
    verbose=1
)

# ================= (OPTIONAL) FINE-TUNING =================
def fine_tune_model(model, base_model):
    print("🔧 Starting fine-tuning...")

    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )

    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[early_stopping_cb, lr_cb],
        class_weight=class_weight_dict,
        verbose=1
    )

    return history_fine

# Uncomment if you want to fine-tune later:
# history_fine = fine_tune_model(model, base_model)

# ================= EVALUATION =================
print("\n" + "="*50)
print("🧠 COMPREHENSIVE EVALUATION")
print("="*50)

best_model = tf.keras.models.load_model(best_model_path)

y_pred_proba = best_model.predict(test_generator)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16 Alzheimer's Classification")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Confidence Analysis
confidences = np.max(y_pred_proba, axis=1)
correct_predictions = (y_pred == y_true)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist([confidences[correct_predictions], confidences[~correct_predictions]],
         bins=20, alpha=0.7, label=['Correct', 'Incorrect'],
         color=['green', 'red'])
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution')
plt.legend()

plt.subplot(1, 3, 2)
class_accuracies = []
for i, class_name in enumerate(class_names):
    mask = (y_true == i)
    accuracy = np.mean(y_pred[mask] == y_true[mask]) if np.sum(mask) > 0 else 0
    class_accuracies.append(accuracy)
plt.bar(class_names, class_accuracies, color='skyblue', alpha=0.7)
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.title('Class-wise Accuracy')
plt.ylim(0, 1)

plt.subplot(1, 3, 3)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training History')
plt.legend()

plt.tight_layout()
plt.show()

# ================= SAVE MODELS =================
print("\n💾 Saving models...")

final_keras = os.path.join(save_dir, "vgg16_adni_final.keras")
best_model.save(final_keras)
print(f"✅ Saved as: {final_keras}")

final_h5 = os.path.join(save_dir, "vgg16_adni_final.h5")
best_model.save(final_h5)
print(f"✅ Saved as: {final_h5}")

final_tf = os.path.join(save_dir, "vgg16_adni_model")
best_model.save(final_tf, save_format='tf')
print(f"✅ Saved as TensorFlow format: {final_tf}")
y
# ================= FINAL SUMMARY =================
print("\n" + "="*50)
print("🎯 TRAINING SUMMARY")
print("="*50)

final_accuracy = np.mean(y_pred == y_true)
print(f"Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
print(f"Total Test Samples: {len(y_true)}")
print(f"Best Model Saved: {best_model_path}")

print("\n✅y Training and evaluation complete!")
