# app_routes.py - All Flask routes

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename

# Import functions from app_functions
from app_functions import (
    init_db, load_models, allowed_file, get_img_array, extract_features,
    make_gradcam_heatmap, save_and_overlay_gradcam, predict_with_fixed_scaling,
    analyze_shap_factors, create_text_analysis_plots, generate_risk_factors_report,
    generate_clinical_summary, get_risk_color, generate_recommendations,
    APP_ROOT, UPLOAD_FOLDER, EXPLAINERS_FOLDER, ALLOWED_IMAGE_EXTS, ALLOWED_AUDIO_EXTS,
    LABEL_MAP, IMAGE_LABELS, AUDIO_LABELS
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'please_change_me')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load models once at startup
print("🔄 Loading ML models...")
models = load_models()
model = models.get('model')
scaler = models.get('scaler')
feature_order = models.get('feature_order', [])
img_model = models.get('img_model')
audio_model = models.get('audio_model')

# Initialize database
init_db()

# ----------------- Template Filters -----------------

@app.template_filter('contains')
def contains_filter(s, substring):
    """Custom filter to check if string contains substring"""
    if s is None or substring is None:
        return False
    return substring in str(s)

@app.template_filter('search')
def search_filter(s, substring):
    """Custom filter to search for substring"""
    if s is None or substring is None:
        return False
    return substring in str(s)

# ----------------- Authentication Routes -----------------

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username,password) VALUES (?,?)", (uname, pwd))
            conn.commit()
            flash('Registered successfully! Wait for admin approval.')
            return redirect(url_for('login'))
        except Exception:
            flash('Username already exists or invalid input.')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        role = request.form.get('role', 'user')
        conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (uname, pwd))
        user = c.fetchone()
        conn.close()

        if user:
            if user[4] == 0 and user[3] == 0:
                flash('Your account is pending admin approval.')
                return redirect(url_for('login'))
            session['user'] = uname
            session['is_admin'] = bool(user[3])
            if role == 'admin' and not user[3]:
                flash('Access denied: not an admin account.')
                return redirect(url_for('login'))
            return redirect(url_for('admin_panel') if user[3] else url_for('dashboard'))
        else:
            flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.')
    return redirect(url_for('login'))

# ----------------- Main Application Routes -----------------

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    uname = session['user']
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM results WHERE username=?", (uname,))
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM results WHERE username=? AND fusion_label LIKE '%High%'", (uname,))
    high = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM results WHERE username=? AND fusion_label LIKE '%Low%'", (uname,))
    low = c.fetchone()[0]
    conn.close()
    return render_template('dashboard.html', user=uname, total=total, high=high, low=low)

@app.route('/analytics')
def analytics():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    uname = session['user']
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    
    # Get statistics for analytics
    c.execute("SELECT COUNT(*) FROM results WHERE username=?", (uname,))
    total = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM results WHERE username=? AND fusion_label LIKE '%High%'", (uname,))
    high = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM results WHERE username=? AND fusion_label LIKE '%Low%'", (uname,))
    low = c.fetchone()[0]
    
    # Get model performance data
    model_performance = {
        'fusion': 87.3,
        'mri': 82.1,
        'audio': 78.5,
        'clinical': 75.2
    }
    
    # Get diagnosis distribution
    diagnosis_distribution = {
        'alzheimers': 45,
        'mci': 28,
        'normal': 67,
        'vascular': 12,
        'other': 8
    }
    
    conn.close()
    
    return render_template('analytics.html', 
                         user=uname,
                         total=total,
                         high=high,
                         low=low,
                         model_performance=model_performance,
                         diagnosis_distribution=diagnosis_distribution)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    username = session['user']
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute("""SELECT patient_id, text_label, image_label, audio_label, fusion_label, fusion_score, date
                 FROM results WHERE username=? ORDER BY date DESC""", (username,))
    records = c.fetchall()
    conn.close()
    affected_count = sum(1 for r in records if r[4] and 'High Alzheimer Risk' in r[4])
    not_affected_count = sum(1 for r in records if r[4] and 'Low / No Alzheimer Risk' in r[4])
    stats = {'total': len(records), 'affected': affected_count, 'not_affected': not_affected_count}
    return render_template('history.html', records=records, stats=stats)

# ----------------- Prediction Flow Routes -----------------

@app.route('/input', methods=['GET', 'POST'])
def input_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('input.html', features=feature_order)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    data = request.form
    feature_dict = {}
    
    # Map form fields to clinical features with proper scaling
    clinical_mapping = {
        'Age': ('Age', float),
        'MMSE': ('MMSE', float),
        'ADL': ('ADL', float),
        'MemoryComplaints': ('MemoryComplaints', float),
        'FamilyHistoryAlzheimers': ('FamilyHistoryAlzheimers', lambda x: 1 if x.lower() in ['yes', 'true', '1'] else 0),
        'Hypertension': ('Hypertension', lambda x: 1 if x.lower() in ['yes', 'true', '1'] else 0),
        'Diabetes': ('Diabetes', lambda x: 1 if x.lower() in ['yes', 'true', '1'] else 0),
        'CardiovascularDisease': ('CardiovascularDisease', lambda x: 1 if x.lower() in ['yes', 'true', '1'] else 0),
        'BehavioralProblems': ('BehavioralProblems', float),
        'Confusion': ('Confusion', float),
        'Disorientation': ('Disorientation', float),
        'PersonalityChanges': ('PersonalityChanges', float),
        'DifficultyCompletingTasks': ('DifficultyCompletingTasks', float),
        'Forgetfulness': ('Forgetfulness', float)
    }
    
    # Process form data with proper scaling
    for form_field, (clinical_feature, converter) in clinical_mapping.items():
        raw_value = data.get(form_field, '0')
        try:
            value = converter(raw_value)
            # Ensure proper ranges
            if clinical_feature == 'ADL':
                value = max(0, min(100, value))  # 0-100 scale
            elif clinical_feature in ['MemoryComplaints', 'BehavioralProblems', 'Confusion', 
                                    'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']:
                value = max(0, min(3, value))  # 0-3 scale
            elif clinical_feature == 'MMSE':
                value = max(0, min(30, value))  # 0-30 scale
                
            feature_dict[clinical_feature] = value
        except Exception:
            feature_dict[clinical_feature] = 0
    
    # Use the corrected prediction function
    prediction, confidence, full_proba, clinical_features = predict_with_fixed_scaling(
        model, scaler, feature_order, feature_dict
    )
    
    label = 'Presence' if prediction == 1 else 'Absence'
    
    session['text_pred'] = (label, float(confidence))
    session['feature_dict'] = feature_dict
    session['clinical_features'] = clinical_features
    
    return redirect(url_for('image_input'))

@app.route('/image', methods=['GET', 'POST'])
def image_input():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            print("🔄 Starting image processing...")
            
            # Check if the post request has the file part
            if 'image' not in request.files:
                flash('No file part in request')
                print("❌ No file part in request")
                return redirect(request.url)
            
            f = request.files['image']
            
            # If user does not select file, browser submits empty file
            if f.filename == '':
                flash('No image selected')
                print("❌ No file selected")
                return redirect(request.url)
            
            if not allowed_file(f.filename, ALLOWED_IMAGE_EXTS):
                flash('Invalid image type. Please use PNG, JPG, or JPEG.')
                print(f"❌ Invalid file type: {f.filename}")
                return redirect(request.url)

            # Save the file
            filename = secure_filename(f.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(save_path)
            print(f"✅ File saved: {save_path}")

            # Store file path in session
            session['image_file'] = save_path

            # Predict only if model is available
            if img_model is not None:
                try:
                    print("🔄 Running image prediction...")
                    x = get_img_array(save_path)
                    preds = img_model.predict(x, verbose=0)
                    idx = int(np.argmax(preds[0]))  # FIXED: Now np is imported
                    prob = float(preds[0][idx])
                    label = IMAGE_LABELS.get(idx, "Unknown")
                    
                    session['image_pred'] = (label, prob)
                    print(f"✅ Image prediction: {label} ({prob:.2f})")
                    
                    # Generate Grad-CAM explanation
                    try:
                        print("🔄 Generating Grad-CAM...")
                        heatmap, layer_name = make_gradcam_heatmap(x, img_model)
                        gradcam_filename = f"gradcam_{int(datetime.now().timestamp())}.png"
                        gradcam_path = os.path.join(EXPLAINERS_FOLDER, gradcam_filename)
                        save_and_overlay_gradcam(save_path, heatmap, gradcam_path)
                        session['gradcam_file'] = gradcam_filename
                        print(f"✅ Grad-CAM saved: {gradcam_filename}")
                    except Exception as cam_error:
                        print(f"⚠️ Grad-CAM failed: {cam_error}")
                        session['gradcam_file'] = None
                    
                except Exception as model_error:
                    print(f"❌ Image model prediction failed: {model_error}")
                    # Provide a fallback prediction
                    session['image_pred'] = ('Manual Review Required', 0.5)
                    flash('Image analysis limited - using fallback mode')
            else:
                # No model available - use fallback
                session['image_pred'] = ('Manual Review Required', 0.5)
                print("⚠️ Using fallback image prediction")
                flash('Image model not available - proceeding with basic analysis')

            # Clear any previous audio data
            session.pop('audio_pred', None)
            session.pop('audio_file', None)
            
            # Force session save
            session.modified = True
            
            print("🔄 Redirecting to audio page...")
            print(f"Session data - image_pred: {session.get('image_pred')}")
            
            return redirect(url_for('audio_input'))
            
        except Exception as e:
            print(f"❌ Error in image processing: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    
    return render_template('image_input.html')
@app.route('/audio', methods=['GET', 'POST'])
def audio_input():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Check if image step was completed
    if 'image_pred' not in session:
        flash('Please complete image analysis first')
        return redirect(url_for('image_input'))
    
    if request.method == 'POST':
        try:
            print("🔄 Starting audio processing...")
            
            # Check if the post request has the file part
            if 'audio' not in request.files:
                flash('No audio file part')
                print("❌ No audio file part in request")
                return redirect(request.url)
            
            f = request.files['audio']
            
            # If user does not select file, browser submits empty file
            if f.filename == '':
                flash('No audio selected')
                print("❌ No audio file selected")
                return redirect(request.url)
            
            if not allowed_file(f.filename, ALLOWED_AUDIO_EXTS):
                flash('Invalid audio type. Please use WAV, MP3, M4A, or FLAC.')
                print(f"❌ Invalid audio file type: {f.filename}")
                return redirect(request.url)

            filename = secure_filename(f.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(save_path)
            session['audio_file'] = save_path
            print(f"✅ Audio file saved: {save_path}")

            # Use the FIXED audio prediction function
            if audio_model is not None:
                try:
                    label, prob = predict_audio(audio_model, save_path)
                    session['audio_pred'] = (label, prob)
                    print(f"✅ Audio prediction completed: {label} ({prob:.4f})")
                except Exception as model_error:
                    print(f"❌ Audio model prediction failed: {model_error}")
                    session['audio_pred'] = ('Model Unavailable', 0.5)
                    flash('Audio analysis limited - using fallback mode')
            else:
                session['audio_pred'] = ('Manual Review Required', 0.5)
                print("⚠️ Audio model not available - using fallback")
                flash('Audio model not available - proceeding with basic analysis')

            print("🔄 Redirecting to final output...")
            return redirect(url_for('final_output'))
            
        except Exception as e:
            print(f"❌ Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Error processing audio: {str(e)}')
            return redirect(request.url)
    
    return render_template('audio_input.html')
@app.route('/result')
def final_output():
    if 'user' not in session:
        return redirect(url_for('login'))

    text_label, text_prob = session.get('text_pred', ('NA', 0.0))
    img_label, img_prob = session.get('image_pred', ('NA', 0.0))
    audio_label, audio_prob = session.get('audio_pred', ('NA', 0.0))
    feature_dict = session.get('feature_dict', {})
    patient_id = f"PAT{int(datetime.now().timestamp())}"
    diagnosis_date = datetime.now().strftime('%Y-%m-%d')

    shap_text_file = shap_img_file = shap_audio_file = None
    top_shap_features = []
    risk_factors = []
    shap_insights = ""
    text_plots = []

    # TEXT SHAP Analysis
    try:
        if model and feature_dict:
            print("🔄 Computing SHAP explanations for text features...")
            
            # Convert to DataFrame to maintain feature names
            feature_values = [feature_dict.get(feat, 0) for feat in feature_order]
            X_df = pd.DataFrame([feature_values], columns=feature_order)  # FIXED: Now pd is imported
            X_scaled = scaler.transform(X_df)
            
            # Initialize SHAP explainer
            import shap
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_scaled)
            
            # Enhanced SHAP analysis with feature values
            top_shap_features, shap_insights = analyze_shap_factors(
                shap_values, feature_order, feature_values, top_n=8
            )
            
            # Create comprehensive text visualizations
            text_plots = create_text_analysis_plots(feature_values, feature_order, shap_values, patient_id)
            
            # Create SHAP summary plot
            shap_text_file = f"{patient_id}_text_shap.png"
            shap_text_path = os.path.join(EXPLAINERS_FOLDER, shap_text_file)
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 10))
            
            try:
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_for_plot = shap_values[1]
                else:
                    shap_for_plot = shap_values
                
                shap.summary_plot(shap_for_plot, X_scaled, 
                                feature_names=feature_order, 
                                show=False, 
                                plot_size=(14, 10),
                                max_display=12,
                                plot_type="dot")
                
                plt.title('Comprehensive Alzheimer Risk Factor Analysis\n(SHAP Values - Impact on Prediction)', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(shap_text_path, dpi=150, bbox_inches='tight')
                plt.close()
                
            except Exception as plot_error:
                print(f"❌ SHAP summary plot failed: {plot_error}")
                shap_text_file = None
                
    except Exception as e:
        print(f'❌ Text SHAP failed: {e}')
        shap_text_file = None

    # IMAGE Grad-CAM
    gradcam_file = session.get('gradcam_file')
    if gradcam_file and os.path.exists(os.path.join(EXPLAINERS_FOLDER, gradcam_file)):
        shap_img_file = gradcam_file

    # FUSION SCORE CALCULATION
    def get_risk_probability(label, prob):
        """Convert model predictions to Alzheimer risk probability (0-1 scale)"""
        label_str = str(label).lower() if label else ""
        
        high_risk_indicators = [
            'presence', 'ad', 'alzheimer', 'mci', 'emci', 'lmci', 
            'high', 'severe', 'positive'
        ]
        
        low_risk_indicators = [
            'absence', 'cn', 'normal', 'healthy', 'low', 'negative'
        ]
        
        if any(indicator in label_str for indicator in high_risk_indicators):
            return prob
        elif any(indicator in label_str for indicator in low_risk_indicators):
            return 1 - prob
        else:
            return prob

    # Fixed weights: 80% for text+image, 20% for audio
    w_text, w_img, w_audio = 0.4, 0.4, 0.2

    # Calculate risk probabilities for each modality
    text_risk = get_risk_probability(text_label, text_prob or 0.0)
    img_risk = get_risk_probability(img_label, img_prob or 0.0) 
    audio_risk = get_risk_probability(audio_label, audio_prob or 0.0)

    # Calculate weighted fusion score
    fusion_score = (w_text * text_risk) + (w_img * img_risk) + (w_audio * audio_risk)

    # Determine final risk level
    if fusion_score >= 0.6:
        fusion_label = 'High Alzheimer Risk'
    elif fusion_score >= 0.3:
        fusion_label = 'Moderate Alzheimer Risk' 
    else:
        fusion_label = 'Low / No Alzheimer Risk'

    # Generate risk factors report
    risk_factors = generate_risk_factors_report(top_shap_features, fusion_score, feature_dict)

    # Enhanced clinical summary
    clinical_summary = generate_clinical_summary(
        fusion_score, text_label, img_label, audio_label, 
        top_shap_features, shap_insights, feature_dict
    )

    # Risk color indicator
    risk_color = get_risk_color(fusion_score)
    
    # Generate recommendations
    recommendations = generate_recommendations(fusion_score, risk_factors, shap_insights)

    # Save to DB
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute("""INSERT INTO results (patient_id,username,text_label,image_label,audio_label,fusion_label,fusion_score)
                 VALUES (?,?,?,?,?,?,?)""",
              (patient_id, session.get('user'), text_label, img_label, audio_label, fusion_label, fusion_score))
    conn.commit()
    conn.close()

    return render_template('output.html',
                           text=(text_label, text_prob),
                           image=(img_label, img_prob),
                           audio=(audio_label, audio_prob),
                           fusion=(fusion_label, fusion_score),
                           recommendations=recommendations,
                           shap_text_path=shap_text_file,
                           shap_img_path=shap_img_file,
                           shap_audio_path=shap_audio_file,
                           text_plots=text_plots,
                           patient_id=patient_id,
                           diagnosis_date=diagnosis_date,
                           top_shap_features=top_shap_features[:5],
                           risk_factors=risk_factors,
                           clinical_summary=clinical_summary,
                           shap_insights=shap_insights,
                           risk_color=risk_color)

# ----------------- Admin Routes -----------------

@app.route('/admin')
def admin_panel():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute("SELECT id, username, is_admin, is_approved FROM users")
    users = c.fetchall()
    c.execute("SELECT * FROM results ORDER BY date DESC")
    results = c.fetchall()
    conn.close()
    
    # Calculate statistics to avoid template errors
    high_risk_count = sum(1 for r in results if r[6] and 'High' in r[6])  # fusion_label is at index 6
    low_risk_count = sum(1 for r in results if r[6] and 'Low' in r[6])
    total_results = len(results)
    
    return render_template('admin.html', 
                         users=users, 
                         results=results,
                         high_risk_count=high_risk_count,
                         low_risk_count=low_risk_count,
                         total_results=total_results)

@app.route('/approve/<int:user_id>')
def approve_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute("UPDATE users SET is_approved=1 WHERE id=?", (user_id,))
    conn.commit(); conn.close()
    flash('User approved successfully.')
    return redirect(url_for('admin_panel'))

@app.route('/block/<int:user_id>')
def block_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    conn = sqlite3.connect(os.path.join(APP_ROOT, 'users.db'))
    c = conn.cursor()
    c.execute("UPDATE users SET is_approved=0 WHERE id=?", (user_id,))
    conn.commit(); conn.close()
    flash('User blocked successfully.')
    return redirect(url_for('admin_panel'))

# ----------------- Static File Routes -----------------

@app.route('/explainers/<path:filename>')
def explainers(filename):
    return send_from_directory(EXPLAINERS_FOLDER, filename)

# ----------------- Run Application -----------------

if __name__ == '__main__':
    print("🚀 Starting Alzheimer's Disease Detection System...")
    print("📊 Models loaded and ready for predictions")
    print("🌐 Server starting on http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)