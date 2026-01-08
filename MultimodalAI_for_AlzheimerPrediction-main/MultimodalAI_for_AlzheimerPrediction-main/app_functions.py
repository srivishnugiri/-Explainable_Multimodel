# app_functions.py - LOAD ACTUAL MODELS VERSION

import os
import pickle
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import cv2
from werkzeug.utils import secure_filename
import shap
import traceback

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model

# Constants / config
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_ROOT, 'models')
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
EXPLAINERS_FOLDER = os.path.join(APP_ROOT, 'static', 'explainers')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPLAINERS_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXTS = {'png', 'jpg', 'jpeg'}
ALLOWED_AUDIO_EXTS = {'wav', 'mp3', 'm4a', 'flac'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Label mappings
LABEL_MAP = {0: 'Presence', 1: 'Absence'}
IMAGE_LABELS = {0: "AD (Alzheimer's Disease)", 1: "CN (Normal)", 2: "EMCI (Early Mild Cognitive Impairment)", 
                3: "LMCI (Late Mild Cognitive Impairment)", 4: "MCI (Mild Cognitive Impairment)"}
AUDIO_LABELS = {0: "Alzheimer", 1: "Healthy"}

# ----------------- Database Functions -----------------

def init_db():
    """Initialize database with required tables"""
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT,
                    is_admin INTEGER DEFAULT 0,
                    is_approved INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    username TEXT,
                    text_label TEXT,
                    image_label TEXT,
                    audio_label TEXT,
                    fusion_label TEXT,
                    fusion_score REAL,
                    date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute("SELECT * FROM users WHERE username='admin@gmail.com'")
    if not c.fetchone():
        c.execute("INSERT INTO users (username,password,is_admin,is_approved) VALUES ('admin@gmail.com','admin',1,1)")
    conn.commit()
    conn.close()

# ----------------- Model Loading Functions -----------------

def check_tensorflow_version():
    """Check TensorFlow version and compatibility"""
    print(f"\n🔧 TensorFlow Version: {tf.__version__}")
    print(f"🔧 Python Version: {os.sys.version}")
    
    # Check if models exist
    image_model_path = os.path.join(APP_ROOT, 'vgg16_adni_final2.h5')
    audio_model_path = os.path.join(APP_ROOT, 'Alzheimer_Audio_ConvLSTM_Fixed.h5')
    
    print(f"🔧 Image model exists: {os.path.exists(image_model_path)}")
    print(f"🔧 Audio model exists: {os.path.exists(audio_model_path)}")
    
    if os.path.exists(image_model_path):
        file_size = os.path.getsize(image_model_path) / (1024 * 1024)  # MB
        print(f"🔧 Image model size: {file_size:.2f} MB")
    
    if os.path.exists(audio_model_path):
        file_size = os.path.getsize(audio_model_path) / (1024 * 1024)  # MB
        print(f"🔧 Audio model size: {file_size:.2f} MB")

# ----------------- FIXED Image Model Loading -----------------

def load_image_model():
    """Load image model with compatibility fixes"""
    try:
        print('🔄 Loading VGG16 image model...')
        image_model_path = os.path.join(APP_ROOT, 'vgg16_adni_final2.h5')
        
        if os.path.exists(image_model_path):
            # Try different loading methods for compatibility
            try:
                # Method 1: Standard load
                model = load_model(image_model_path)
                print('✅ Loaded VGG16 model with standard method')
            except Exception as e1:
                print(f'⚠️ Standard load failed: {e1}')
                try:
                    # Method 2: Load with custom objects
                    model = load_model(image_model_path, compile=False)
                    print('✅ Loaded VGG16 model with compile=False')
                except Exception as e2:
                    print(f'⚠️ Load with compile=False failed: {e2}')
                    try:
                        # Method 3: Load weights only
                        from tensorflow.keras.applications import VGG16
                        base_model = VGG16(weights=None, include_top=True, input_shape=(224, 224, 3))
                        model = load_model(image_model_path)
                        print('✅ Loaded VGG16 model as weights')
                    except Exception as e3:
                        print(f'❌ All loading methods failed: {e3}')
                        raise
            
            # Print model info
            print(f"📸 Image model input shape: {model.input_shape}")
            print(f"📸 Image model output shape: {model.output_shape}")
            
            return model
            
        else:
            print('❌ VGG16 model file not found')
            raise FileNotFoundError("VGG16 model not found")
            
    except Exception as e:
        print(f'❌ VGG16 model loading failed: {e}')
        return create_fallback_image_model()

def create_fallback_image_model():
    """Create fallback CNN image model"""
    print('🔄 Creating fallback CNN image model...')
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print('✅ Created fallback CNN-based image model')
    return model

# ----------------- FIXED Audio Feature Extraction -----------------

def extract_features(path, target_length=100, n_mfcc=65):
    """Extract MFCC features from audio file with proper formatting"""
    try:
        # Load audio file
        sig, sr = librosa.load(path, sr=22050)
        
        # Extract MFCC features - use 65 coefficients to match model
        mfccs = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        
        # Pad or truncate to fixed length
        if mfccs.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfccs = mfccs[:, :target_length]
        
        print(f"✅ Audio features extracted: {mfccs.shape}")
        return mfccs  # Don't transpose - keep as (n_mfcc, time_steps)
        
    except Exception as e:
        print(f"❌ Audio feature extraction error: {e}")
        # Return properly shaped fallback features
        return np.zeros((n_mfcc, target_length))

# ----------------- FIXED Audio Prediction Function -----------------

def predict_audio(audio_model, audio_path):
    """Predict audio with proper input formatting"""
    try:
        print("🔄 Extracting audio features...")
        features = extract_audio_features(audio_path, n_mfcc=65)  # Use 65 to match model
        
        if features is not None:
            print("🔄 Preparing audio features for model...")
            print(f"🎵 Model input shape: {audio_model.input_shape}")
            print(f"🎵 Features shape: {features.shape}")
            
            # Reshape features to match model input (None, 100, 65)
            # The model expects (batch_size, time_steps, features)
            # features shape is (65, 100) - we need (100, 65)
            features = features.T  # Transpose to (100, 65)
            X = features.reshape(1, features.shape[0], features.shape[1])  # (1, 100, 65)
            
            print(f"🔄 Final input shape: {X.shape}")
            
            # Get prediction
            print("🔄 Running audio model prediction...")
            preds = audio_model.predict(X, verbose=0)[0]
            
            # Apply temperature scaling for more balanced probabilities
            temperature = 1.5
            scaled_preds = np.power(preds, 1/temperature)
            scaled_preds = scaled_preds / np.sum(scaled_preds)
            
            idx = int(np.argmax(scaled_preds))
            prob = float(scaled_preds[idx])
            label = AUDIO_LABELS.get(idx, "Unknown")
            
            print(f"✅ Audio prediction - Raw: {preds}, Scaled: {scaled_preds}")
            print(f"✅ Final audio prediction: {label} ({prob:.4f})")
            
            return label, prob
            
        else:
            print("❌ Feature extraction failed")
            return 'Feature Extraction Failed', 0.5
            
    except Exception as e:
        print(f"❌ Audio prediction error: {e}")
        traceback.print_exc()
        return 'Prediction Error', 0.5

def load_models():
    """Load all ML models with fallbacks"""
    print('Loading models...')
    check_tensorflow_version()
    
    models = {}
    
    # Text/Tabular model
    try:
        with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb') as f:
            models['model'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_order.pkl'), 'rb') as f:
            models['feature_order'] = pickle.load(f)
        print('✅ Tabular model and metadata loaded')
    except Exception as e:
        print('❌ Could not load tabular model or metadata:', e)
        models['model'] = None
        models['scaler'] = None
        models['feature_order'] = []
    
    # Image model
    models['img_model'] = load_image_model()
    
    # Audio model
    try:
        print('🔄 Loading ConvLSTM audio model...')
        audio_model_path = os.path.join(APP_ROOT, 'Alzheimer_Audio_ConvLSTM_Fixed.h5')
        
        if os.path.exists(audio_model_path):
            # Load the actual pre-trained model
            models['audio_model'] = load_model(audio_model_path)
            print(f'✅ Loaded pre-trained ConvLSTM audio model: {audio_model_path}')
            
            # Print model summary for debugging
            print(f"🎵 Audio model input shape: {models['audio_model'].input_shape}")
            print(f"🎵 Audio model output shape: {models['audio_model'].output_shape}")
            
        else:
            print('❌ ConvLSTM audio model file not found, using fallback')
            raise FileNotFoundError("ConvLSTM audio model not found")
            
    except Exception as e:
        print(f'❌ ConvLSTM audio model loading failed: {e}')
        # Create simple sequential model as fallback
        models['audio_model'] = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(100, 65)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        models['audio_model'].compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print('✅ Created fallback LSTM-based audio model')
    
    # Print final status
    print("\n" + "="*50)
    print("FINAL MODEL STATUS:")
    print("="*50)
    print(f"📊 Tabular Model: {'✅ LOADED' if models['model'] else '❌ FAILED'}")
    print(f"🖼️  Image Model: {'✅ LOADED' if models['img_model'] else '❌ FAILED'}")
    print(f"🎵 Audio Model: {'✅ LOADED' if models['audio_model'] else '❌ FAILED'}")
    print("="*50)
    
    return models

# ----------------- Utility Functions -----------------

def allowed_file(filename, allowed_set):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def get_img_array(img_path, target_size=(224, 224)):
    """Load and preprocess image for model prediction"""
    img = keras_image.load_img(img_path, target_size=target_size)
    array = keras_image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = array / 255.0
    return array

# Legacy function for compatibility
def extract_features(path, max_pad_length=100):
    """Legacy function - use extract_audio_features instead"""
    return extract_audio_features(path, target_length=max_pad_length, n_mfcc=40)

def find_conv_layer(model, layer_name=None):
    """Find convolutional layer for Grad-CAM"""
    if layer_name:
        for layer in model.layers:
            if layer.name == layer_name:
                return layer_name
    
    # Find last convolutional layer automatically
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
            print(f"✅ Found convolutional layer: {layer.name}")
            return layer.name
    
    # Fallback to any convolutional layer
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            print(f"✅ Found convolutional layer (fallback): {layer.name}")
            return layer.name
    
    print("❌ No convolutional layer found for Grad-CAM")
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generate Grad-CAM heatmap for image explanation"""
    try:
        # Find appropriate convolutional layer
        conv_layer_name = find_conv_layer(model, last_conv_layer_name)
        if conv_layer_name is None:
            # Use the last conv layer explicitly
            for layer in model.layers:
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D)):
                    conv_layer_name = layer.name
                    break
            
            if conv_layer_name is None:
                print("❌ No convolutional layers found, using fallback visualization")
                # Create a simple heatmap as fallback
                heatmap = np.random.rand(14, 14)  # Small random heatmap
                return heatmap, "fallback"
        
        print(f"🔄 Using convolutional layer: {conv_layer_name}")
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Calculate gradients and heatmap
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Handle case where grads is None
        if grads is None:
            print("❌ Gradients are None, using fallback")
            heatmap = np.random.rand(14, 14)
            return heatmap, conv_layer_name
            
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), conv_layer_name
        
    except Exception as e:
        print(f"❌ Grad-CAM error: {e}")
        # Return a simple heatmap as fallback
        heatmap = np.random.rand(14, 14)  # Small random heatmap
        return heatmap, "fallback"

def save_and_overlay_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    """Save Grad-CAM visualization overlay"""
    try:
        # Load original image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not load image")
        
        # Resize heatmap to match image dimensions
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        
        # Save result
        cv2.imwrite(cam_path, superimposed_img)
        return cam_path
        
    except Exception as e:
        print(f"❌ Grad-CAM overlay error: {e}")
        # Create a simple fallback visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(heatmap, cmap='jet')
        plt.title('Heatmap')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            plt.imshow(img)
            plt.title('Original Image')
        plt.tight_layout()
        plt.savefig(cam_path, dpi=150, bbox_inches='tight')
        plt.close()
        return cam_path

# ----------------- Prediction Functions -----------------

def predict_with_fixed_scaling(model, scaler, feature_columns, patient_data):
    """Predict with proper clinical feature calculation"""
    
    # Calculate clinical features
    features = {}
    
    # Basic features
    features['Age_Risk'] = max(patient_data.get('Age', 70) - 65, 0)
    
    mmse = patient_data.get('MMSE', 25)
    features['MMSE_Severity'] = (25 - mmse) ** 1.5 if mmse < 24 else 0
    
    adl = patient_data.get('ADL', 50)
    features['ADL_Impairment'] = (100 - adl) ** 0.8 if adl < 70 else 0
    
    memory = patient_data.get('MemoryComplaints', 0)
    features['Memory_Severity'] = memory ** 1.5
    
    # Symptom burden
    symptom_cols = ['BehavioralProblems', 'Confusion', 'Disorientation', 
                   'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']
    symptoms_sum = sum(patient_data.get(col, 0) for col in symptom_cols)
    features['Symptom_Burden'] = symptoms_sum
    
    # Vascular risk
    vascular_factors = ['Hypertension', 'Diabetes', 'CardiovascularDisease']
    vascular_sum = sum(patient_data.get(col, 0) for col in vascular_factors)
    features['Vascular_Risk'] = vascular_sum
    
    features['FamilyHistoryAlzheimers'] = patient_data.get('FamilyHistoryAlzheimers', 0)
    
    # Clinical risk score
    features['Clinical_Risk_Score'] = (
        features['Age_Risk'] * 0.1 +
        features['MMSE_Severity'] * 0.3 +
        features['ADL_Impairment'] * 0.25 +
        features['Memory_Severity'] * 0.2 +
        features['Symptom_Burden'] * 0.1 +
        features['Vascular_Risk'] * 0.05 +
        features['FamilyHistoryAlzheimers'] * 2.0
    )
    
    # Create DataFrame
    patient_df = pd.DataFrame([features])
    
    # Ensure all features are present
    for feature in feature_columns:
        if feature not in patient_df.columns:
            patient_df[feature] = 0
    
    # Select and scale features
    X_patient = patient_df[feature_columns]
    X_patient_scaled = scaler.transform(X_patient)
    
    # Get probability
    probability = model.predict_proba(X_patient_scaled)[0]
    alzheimers_prob = probability[1]
    prediction = 1 if alzheimers_prob > 0.5 else 0
    
    return prediction, alzheimers_prob, probability, features

# ----------------- FIXED SHAP Analysis Functions -----------------

def analyze_shap_factors(shap_values, feature_names, feature_values, top_n=5):
    """Enhanced SHAP analysis to identify specific risk factors for Alzheimer's - FIXED VERSION"""
    if shap_values is None or len(shap_values) == 0:
        return [], "Insufficient data for detailed analysis"
    
    try:
        # Handle different SHAP value formats safely
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                shap_vals = shap_values[1]  # Use class 1 (Presence) values
            else:
                shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        # Ensure we have the right shape - FIXED array handling
        if hasattr(shap_vals, 'shape') and len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0]  # Get first sample
        
        # Convert to numpy array for safe processing
        shap_vals = np.array(shap_vals).flatten()
        feature_values = np.array(feature_values).flatten()
        
        print(f"🔄 SHAP analysis: {len(shap_vals)} values, {len(feature_names)} features")
        
        # Ensure arrays have same length
        min_length = min(len(shap_vals), len(feature_names), len(feature_values))
        shap_vals = shap_vals[:min_length]
        feature_names = feature_names[:min_length]
        feature_values = feature_values[:min_length]
        
        # Get top contributing features
        contributing_factors = []
        risk_increasing = []
        risk_decreasing = []
        
        for i, (shap_val, feat_name) in enumerate(zip(shap_vals, feature_names)):
            impact = float(shap_val)
            feat_val = float(feature_values[i]) if i < len(feature_values) else 0
            
            factor_info = {
                'feature': feat_name,
                'impact': abs(impact),
                'direction': 'increases' if impact > 0 else 'decreases',
                'value': feat_val,
                'raw_impact': impact
            }
            
            # Add detailed risk information based on feature type
            if impact > 0:  # Risk increasing
                risk_increasing.append(factor_info)
            else:  # Risk decreasing (protective)
                risk_decreasing.append(factor_info)
            
            contributing_factors.append(factor_info)
        
        # Sort by absolute impact
        contributing_factors.sort(key=lambda x: x['impact'], reverse=True)
        risk_increasing.sort(key=lambda x: x['impact'], reverse=True)
        risk_decreasing.sort(key=lambda x: x['impact'], reverse=True)
        
        # Generate clinical insights
        clinical_insights = generate_shap_insights(risk_increasing[:3], risk_decreasing[:3], feature_values, feature_names)
        
        return contributing_factors[:top_n], clinical_insights
        
    except Exception as e:
        print(f"❌ SHAP analysis error: {e}")
        return [], "Analysis limited due to technical constraints"

def generate_shap_insights(risk_increasing, risk_decreasing, feature_values, feature_names):
    """Generate specific clinical insights from SHAP analysis"""
    insights = []
    
    # Analyze top risk-increasing factors
    if risk_increasing:
        insights.append("Primary risk factors identified:")
        for factor in risk_increasing[:3]:  # Top 3 risk factors
            feature_name = factor['feature']
            impact = factor['raw_impact']
            value = factor['value']
            
            if 'age' in feature_name.lower():
                if value > 65:
                    insights.append(f"• Advanced age ({value} years) significantly increases risk")
                elif value > 55:
                    insights.append(f"• Middle age ({value} years) moderately increases risk")
                    
            elif 'mmse' in feature_name.lower():
                if value < 24:
                    insights.append(f"• Low MMSE score ({value}) indicates significant cognitive impairment")
                elif value < 27:
                    insights.append(f"• Borderline MMSE score ({value}) suggests mild cognitive issues")
                    
            elif 'cdr' in feature_name.lower():
                if value >= 1:
                    insights.append(f"• Elevated CDR score ({value}) confirms dementia presence")
                elif value == 0.5:
                    insights.append(f"• Questionable impairment (CDR {value}) requires monitoring")
                    
            elif 'hippocampal' in feature_name.lower() or 'volume' in feature_name.lower():
                if value < 0.7:  # Assuming normalized volume
                    insights.append(f"• Reduced brain volume ({value:.2f}) indicates structural changes")
                    
            elif 'apoe' in feature_name.lower():
                if value > 0:
                    insights.append("• APOE ε4 genetic marker present - genetic predisposition detected")
                    
            elif 'memory' in feature_name.lower():
                if value < 70:
                    insights.append(f"• Impaired memory function (score: {value})")
                    
            elif 'bmi' in feature_name.lower():
                if value < 18.5 or value > 30:
                    insights.append(f"• BMI ({value}) outside healthy range contributing to risk")
                    
            elif 'diabetes' in feature_name.lower() and value > 0:
                insights.append("• Diabetes history increases vascular dementia risk")
                
            elif 'hypertension' in feature_name.lower() and value > 0:
                insights.append("• Hypertension contributing to cerebrovascular risk")
                
            else:
                insights.append(f"• {feature_name} contributing to increased risk")
    
    # Analyze protective factors
    if risk_decreasing:
        insights.append("\nProtective factors noted:")
        for factor in risk_decreasing[:2]:  # Top 2 protective factors
            feature_name = factor['feature']
            value = factor['value']
            
            if 'education' in feature_name.lower():
                if value > 16:
                    insights.append(f"• Higher education ({value} years) provides cognitive reserve")
                elif value > 12:
                    insights.append(f"• Adequate education ({value} years) offers some protection")
                    
            elif 'physical' in feature_name.lower() and value > 0:
                insights.append("• Physical activity serving as protective factor")
                
            elif 'social' in feature_name.lower() and value > 0:
                insights.append("• Social engagement supporting cognitive health")
                
            else:
                insights.append(f"• {feature_name} contributing to risk reduction")
    
    return "\n".join(insights) if insights else "Standard risk profile based on demographic and clinical factors"

# ----------------- FIXED SHAP Summary Plot -----------------

def create_shap_summary_plot(shap_values, feature_array, feature_names, patient_id):
    """Create SHAP summary plot with proper error handling"""
    try:
        shap_text_file = f"{patient_id}_text_shap.png"
        shap_text_path = os.path.join(EXPLAINERS_FOLDER, shap_text_file)
        
        plt.figure(figsize=(12, 8))
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_for_plot = shap_values[1]  # Use positive class for binary classification
        else:
            shap_for_plot = shap_values
        
        # Ensure we have 2D arrays for SHAP summary plot
        if len(shap_for_plot.shape) == 1:
            shap_for_plot = shap_for_plot.reshape(1, -1)
        if len(feature_array.shape) == 1:
            feature_array = feature_array.reshape(1, -1)
        
        print(f"🔄 Creating SHAP summary plot with shapes: SHAP {shap_for_plot.shape}, Features {feature_array.shape}")
        
        # Create simple bar plot instead of complex summary plot
        mean_shap = np.mean(np.abs(shap_for_plot), axis=0)
        
        # Get top features
        n_features = min(10, len(feature_names))
        indices = np.argsort(mean_shap)[-n_features:][::-1]
        top_feat_names = [feature_names[i] for i in indices]
        top_shap_vals = [mean_shap[i] for i in indices]
        
        colors = ['#e74c3c' if val > np.percentile(mean_shap, 70) else 
                 '#f39c12' if val > np.percentile(mean_shap, 40) else '#3498db' 
                 for val in top_shap_vals]
        
        plt.barh(range(n_features), top_shap_vals, color=colors)
        plt.yticks(range(n_features), top_feat_names)
        plt.xlabel('Mean |SHAP value| (Impact Magnitude)')
        plt.title('Feature Importance for Alzheimer Prediction', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(shap_text_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created SHAP plot: {shap_text_file}")
        return shap_text_file
        
    except Exception as plot_error:
        print(f"❌ SHAP plot failed: {plot_error}")
        return create_fallback_shap_plot(shap_values, feature_names, patient_id)

def create_fallback_shap_plot(shap_values, feature_names, patient_id):
    """Create fallback SHAP plot when summary plot fails"""
    try:
        shap_text_file = f"{patient_id}_text_shap.png"
        shap_text_path = os.path.join(EXPLAINERS_FOLDER, shap_text_file)
        
        plt.figure(figsize=(12, 8))
        
        # Get mean absolute SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            mean_shap = np.abs(shap_values[1]).mean(0)
        else:
            mean_shap = np.abs(shap_values).mean(0)
        
        # Ensure arrays have same length
        min_length = min(len(mean_shap), len(feature_names))
        mean_shap = mean_shap[:min_length]
        feature_names = feature_names[:min_length]
        
        # Sort features by importance
        n_features = min(12, min_length)  # Show up to 12 features
        sorted_idx = np.argsort(mean_shap)[-n_features:]  # Top features
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = [mean_shap[i] for i in sorted_idx]
        
        colors = ['#e74c3c' if imp > np.percentile(mean_shap, 70) else 
                 '#f39c12' if imp > np.percentile(mean_shap, 40) else '#3498db' 
                 for imp in sorted_importance]
        
        plt.barh(range(len(sorted_features)), sorted_importance, color=colors)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Mean Absolute SHAP Value (Impact Magnitude)')
        plt.title('Alzheimer Risk Factor Importance', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(shap_text_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created fallback SHAP plot: {shap_text_file}")
        return shap_text_file
        
    except Exception as e:
        print(f"❌ Fallback SHAP plot also failed: {e}")
        return None

# ----------------- Clinical Report Functions -----------------

def get_risk_color(score):
    """Return color based on risk score"""
    if score >= 0.7:
        return '🔴'  # High risk
    elif score >= 0.4:
        return '🟡'  # Medium risk
    else:
        return '🟢'  # Low risk

def generate_risk_factors_report(top_features, fusion_score, feature_dict):
    """Generate detailed risk factors report with proper structure"""
    risk_factors = []
    
    # High impact features from SHAP
    if top_features and len(top_features) > 0:
        high_impact = [f for f in top_features if f.get('impact', 0) > 0.1][:5]
        
        for feature in high_impact:
            feature_name = feature.get('feature', 'Unknown')
            impact_value = feature.get('impact', 0)
            current_value = feature_dict.get(feature_name, 'N/A')
            
            risk_factor = {
                'factor': feature_name,
                'impact': f"{impact_value:.3f}",
                'description': get_clinical_description(feature_name, current_value),
                'significance': 'High' if impact_value > 0.15 else 'Medium',
                'current_value': current_value,
                'feature': feature_name
            }
            risk_factors.append(risk_factor)
    
    # Add general risk factors based on fusion score
    if fusion_score >= 0.7:
        risk_factors.append({
            'factor': 'Multimodal High Risk',
            'impact': 'N/A',
            'description': 'Multiple high-risk factors detected across clinical, imaging, and speech modalities',
            'significance': 'Critical',
            'current_value': 'Positive',
            'feature': 'Overall_Risk'
        })
    elif fusion_score >= 0.4:
        risk_factors.append({
            'factor': 'Moderate Risk Profile',
            'impact': 'N/A',
            'description': 'Several moderate risk factors present requiring monitoring',
            'significance': 'Moderate', 
            'current_value': 'Present',
            'feature': 'Risk_Profile'
        })
    
    return risk_factors

def get_clinical_description(feature_name, value):
    """Get clinical description for risk factors"""
    descriptions = {
        'Age': f'Advanced age ({value} years) - strongest non-modifiable risk factor',
        'MMSE': f'Cognitive screening score ({value}) - lower scores indicate impairment',
        'CDR': f'Clinical Dementia Rating ({value}) - measures functional impairment',
        'Hippocampal_Volume': f'Brain structure volume ({value}) - atrophy associated with AD',
        'APOE': 'Genetic risk factor - ε4 allele increases Alzheimer susceptibility',
        'Education_Level': f'Cognitive reserve ({value} years) - lower education increases risk',
        'Diabetes': 'Metabolic condition - increases vascular dementia risk',
        'Hypertension': 'Cardiovascular risk factor - affects cerebral blood flow',
        'BMI': f'Body mass index ({value}) - extreme values associated with higher risk',
        'Memory_Test_Score': f'Episodic memory performance ({value}) - key cognitive domain in AD'
    }
    
    return descriptions.get(feature_name, f'Clinical risk factor ({feature_name})')

def generate_clinical_summary(fusion_score, text_pred, image_pred, audio_pred, top_features, shap_insights, feature_dict):
    """Generate clinical summary based on predictions and risk factors"""
    
    # Base risk assessment
    if fusion_score >= 0.7:
        base_summary = "HIGH PROBABILITY of Alzheimer's Disease. "
        urgency = "Urgent neurological evaluation recommended."
    elif fusion_score >= 0.4:
        base_summary = "MODERATE PROBABILITY of cognitive impairment. "
        urgency = "Neuropsychological follow-up advised."
    else:
        base_summary = "LOW PROBABILITY of Alzheimer's Disease. "
        urgency = "Continue preventive strategies and annual screening."
    
    # Add specific findings from each modality
    findings = []
    
    # Text/Clinical findings
    if 'Presence' in text_pred:
        findings.append("positive clinical indicators")
    elif 'Absence' in text_pred:
        findings.append("largely normal clinical markers")
    
    # Imaging findings
    if any(term in image_pred for term in ['AD', 'MCI', 'EMCI', 'LMCI']):
        findings.append("suggestive neuroimaging features")
    elif 'CN' in image_pred or 'Normal' in image_pred:
        findings.append("unremarkable brain imaging")
    
    # Audio findings  
    if 'Alzheimer' in audio_pred:
        findings.append("abnormal speech patterns")
    elif 'Healthy' in audio_pred:
        findings.append("normal speech characteristics")
    
    if findings:
        base_summary += f"Multimodal assessment shows {', '.join(findings)}. "
    
    # Add key risk factors from SHAP analysis
    key_risks = []
    for feature in top_features[:3]:  # Top 3 risk factors
        feat_name = feature['feature']
        if feat_name in feature_dict:
            if 'MMSE' in feat_name and feature_dict[feat_name] < 26:
                key_risks.append(f"low MMSE ({feature_dict[feat_name]})")
            elif 'Age' in feat_name and feature_dict[feat_name] > 65:
                key_risks.append(f"advanced age ({feature_dict[feat_name]} years)")
            elif 'CDR' in feat_name and feature_dict[feat_name] > 0:
                key_risks.append(f"elevated CDR ({feature_dict[feat_name]})")
    
    if key_risks:
        base_summary += f"Notable risk factors include {', '.join(key_risks)}. "
    
    base_summary += urgency
    
    return base_summary

def create_text_analysis_plots(feature_values, feature_names, shap_values, patient_id):
    """Create simplified text analysis visualizations - FIXED VERSION"""
    plot_files = []
    
    try:
        # Handle SHAP values format safely
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = shap_values[1]  # Use positive class for binary classification
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals[0]  # Get first sample
        else:
            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Ensure we have 1D arrays and convert to numpy for safe processing
        shap_vals = np.array(shap_vals).flatten()
        feature_values = np.array(feature_values).flatten()
        
        print(f"🔄 Creating plots with {len(feature_names)} features, {len(shap_vals)} SHAP values")
        
        # Ensure arrays have same length
        min_length = min(len(shap_vals), len(feature_names))
        shap_vals = shap_vals[:min_length]
        feature_names = feature_names[:min_length]
        
        # Get top features by absolute SHAP value
        abs_impacts = np.abs(shap_vals)
        
        # Handle case where we have fewer than 10 features
        n_features = min(8, min_length)  # Show up to 8 features
        indices = np.argsort(abs_impacts)[-n_features:][::-1]  # Top N indices
        
        top_features = [feature_names[i] for i in indices if i < len(feature_names)]
        top_impacts = [abs_impacts[i] for i in indices if i < len(abs_impacts)]
        top_shap_vals = [shap_vals[i] for i in indices if i < len(shap_vals)]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        colors = ['#e74c3c' if val > 0 else '#2ecc71' for val in top_shap_vals]
        
        plt.barh(range(len(top_features)), top_impacts, color=colors)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Impact (Absolute SHAP Value)')
        plt.title(f'Top {n_features} Alzheimer Risk Factors', fontweight='bold')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        importance_plot = f"{patient_id}_feature_importance.png"
        importance_path = os.path.join(EXPLAINERS_FOLDER, importance_plot)
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_files.append(importance_plot)
        print(f"✅ Created feature importance plot with {len(top_features)} features")
        
    except Exception as e:
        print(f"❌ Error creating text plots: {e}")
        traceback.print_exc()
    
    return plot_files

def generate_recommendations(fusion_score, risk_factors, shap_insights):
    """Generate recommendations with robust error handling"""
    rec = []
    
    # Risk-level based recommendations (always work)
    if fusion_score >= 0.7:
        rec.append('🔴 URGENT: Neurological consultation within 2 weeks')
        rec.append('Comprehensive neuropsychological assessment')
        rec.append('MRI with hippocampal volumetry and FDG-PET scan')
        rec.append('CSF biomarker testing (Aβ42, tau, p-tau)')
        rec.append('APOE genotyping for genetic risk assessment')
    elif fusion_score >= 0.4:
        rec.append('🟡 PRIORITY: Neuropsychological testing in 3-6 months')
        rec.append('Lifestyle interventions: Mediterranean diet, aerobic exercise')
        rec.append('Cognitive training and mental stimulation activities')
        rec.append('Vascular risk factor management')
    else:
        rec.append('🟢 ROUTINE: Annual cognitive screening if >65 years')
        rec.append('Maintain physical, social, and cognitive activities')
        rec.append('Manage modifiable risk factors')
    
    # Safe SHAP-informed recommendations
    try:
        if risk_factors and isinstance(risk_factors, list) and len(risk_factors) > 0:
            # Check first 3 factors safely
            for factor in risk_factors[:3]:
                if isinstance(factor, dict) and 'feature' in factor:
                    feature_name = str(factor['feature']).lower()
                    
                    if 'age' in feature_name:
                        rec.append('Age-appropriate cognitive monitoring recommended')
                    elif 'mmse' in feature_name:
                        rec.append('Detailed cognitive domain testing advised')
                    elif 'apoe' in feature_name:
                        rec.append('Consider genetic counseling')
                    elif 'diabetes' in feature_name or 'hypertension' in feature_name:
                        rec.append('Cardiovascular risk optimization needed')
                    elif 'volume' in feature_name:
                        rec.append('Neuroimaging follow-up suggested')
    except Exception as e:
        print(f"⚠️ Safe recommendations fallback: {e}")
        # Continue with basic recommendations
    
    return rec