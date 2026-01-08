import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_model_accuracy_plot():
    """Create Model Accuracy vs Energy Consumption plot"""
    plt.figure(figsize=(10, 6))
    
    # Data from the first image
    models = ['ANN (GPU)', 'ANN (CPU)', 'SNN (Loihi)', 'SNN (TrueNorth)']
    accuracy = [91, 89, 88, 85]  # Approximate values from image
    energy_consumption = [-160, -140, -120, -100]  # Approximate values
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Create scatter plot
    scatter = plt.scatter(energy_consumption, accuracy, s=200, c=colors, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add labels and annotations
    for i, model in enumerate(models):
        plt.annotate(model, (energy_consumption[i], accuracy[i]), 
                    xytext=(10, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    plt.xlabel('Energy Consumption per Inference', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Accuracy vs. Energy Consumption per Inference', fontsize=14, fontweight='bold')
    
    # Customize grid and limits
    plt.grid(True, alpha=0.3)
    plt.xlim(-170, -90)
    plt.ylim(82, 95)
    
    # Add accuracy percentage labels
    for i, acc in enumerate(accuracy):
        plt.text(energy_consumption[i], acc-1, f'{acc}%', 
                ha='center', va='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_accuracy_vs_energy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'model_accuracy_vs_energy.png'

def create_power_output_plot():
    """Create Actual vs Predicted Power Output plot"""
    plt.figure(figsize=(12, 6))
    
    # Sample time series data
    time_steps = np.arange(0, 26, 5)
    actual_power = [120, 115, 105, 95, 85, 80]  # Sample data
    predicted_power = [118, 112, 100, 92, 83, 78]  # Sample data
    
    # Create line plot
    plt.plot(time_steps, actual_power, 'o-', linewidth=3, markersize=8, 
             label='Actual Power Output', color='#2E86AB', alpha=0.8)
    plt.plot(time_steps, predicted_power, 's--', linewidth=2, markersize=6,
             label='Predicted Power Output (LSTM)', color='#A23B72', alpha=0.8)
    
    plt.xlabel('Time Steps', fontsize=12, fontweight='bold')
    plt.ylabel('Power Output (kW)', fontsize=12, fontweight='bold')
    plt.title('Actual vs Predicted Power Output', fontsize=14, fontweight='bold')
    
    # Customize grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set x-ticks explicitly
    plt.xticks(time_steps)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'power_output_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'power_output_comparison.png'

def create_model_performance_comparison():
    """Create comprehensive model performance comparison"""
    plt.figure(figsize=(14, 10))
    
    # Model names
    models = ['Fusion Model', 'MRI CNN', 'Audio Model', 'Clinical RF', 'ANN GPU', 'ANN CPU', 'SNN Loihi', 'SNN TrueNorth']
    
    # Performance metrics (sample data)
    accuracy = [87.3, 82.1, 78.5, 75.2, 91.0, 89.0, 88.0, 85.0]
    precision = [85.1, 80.3, 76.8, 73.5, 89.5, 87.2, 86.1, 83.2]
    recall = [86.8, 81.5, 77.9, 74.8, 90.2, 88.1, 87.3, 84.1]
    f1_score = [85.9, 80.9, 77.3, 74.1, 89.8, 87.6, 86.7, 83.6]
    
    x_pos = np.arange(len(models))
    width = 0.2
    
    # Create grouped bar chart
    plt.bar(x_pos - 1.5*width, accuracy, width, label='Accuracy', color='#3498DB', alpha=0.8)
    plt.bar(x_pos - 0.5*width, precision, width, label='Precision', color='#2ECC71', alpha=0.8)
    plt.bar(x_pos + 0.5*width, recall, width, label='Recall', color='#E74C3C', alpha=0.8)
    plt.bar(x_pos + 1.5*width, f1_score, width, label='F1-Score', color='#F39C12', alpha=0.8)
    
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, models, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, acc in enumerate(accuracy):
        plt.text(i - 1.5*width, acc + 1, f'{acc}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'model_performance_comparison.png'

def create_confusion_matrix_plot():
    """Create confusion matrix for model evaluation"""
    plt.figure(figsize=(10, 8))
    
    # Sample confusion matrix data
    cm_data = np.array([[65, 5, 3, 2, 1],
                       [4, 58, 6, 3, 2],
                       [2, 4, 52, 8, 3],
                       [1, 3, 7, 48, 5],
                       [0, 2, 4, 6, 42]])
    
    classes = ['AD', 'CN', 'EMCI', 'LMCI', 'MCI']
    
    # Create heatmap
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Number of Cases'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Alzheimer Disease Classification', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'confusion_matrix.png'

def create_roc_curves():
    """Create ROC curves for different models"""
    plt.figure(figsize=(10, 8))
    
    # Sample ROC data
    fpr_fusion = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr_fusion = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9, 0.94, 0.97, 0.99, 1.0])
    
    fpr_mri = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr_mri = np.array([0.0, 0.15, 0.35, 0.55, 0.7, 0.8, 0.87, 0.92, 0.95, 0.98, 1.0])
    
    fpr_audio = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tpr_audio = np.array([0.0, 0.1, 0.3, 0.5, 0.65, 0.75, 0.83, 0.89, 0.93, 0.96, 1.0])
    
    # Plot ROC curves
    plt.plot(fpr_fusion, tpr_fusion, linewidth=3, label='Fusion Model (AUC = 0.94)', color='#E74C3C')
    plt.plot(fpr_mri, tpr_mri, linewidth=2, label='MRI Model (AUC = 0.89)', color='#3498DB')
    plt.plot(fpr_audio, tpr_audio, linewidth=2, label='Audio Model (AUC = 0.85)', color='#2ECC71')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'roc_curves.png'

def create_training_history():
    """Create training and validation history plot"""
    plt.figure(figsize=(12, 8))
    
    # Sample training history
    epochs = range(1, 51)
    train_loss = [0.8, 0.6, 0.45, 0.35, 0.28, 0.23, 0.19, 0.16, 0.14, 0.12] + \
                [0.11 - i*0.002 for i in range(40)]
    val_loss = [0.82, 0.65, 0.5, 0.4, 0.33, 0.28, 0.24, 0.21, 0.19, 0.17] + \
              [0.16 - i*0.001 for i in range(40)]
    train_acc = [65, 72, 78, 83, 86, 88, 90, 91, 92, 93] + \
               [93 + i*0.1 for i in range(40)]
    val_acc = [63, 70, 76, 80, 83, 85, 87, 88, 89, 90] + \
             [90 + i*0.08 for i in range(40)]
    
    # Create subplots
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training and Validation History', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'training_history.png'

def create_feature_importance_plot():
    """Create feature importance plot for clinical data"""
    plt.figure(figsize=(12, 8))
    
    # Sample feature importance data
    features = ['MMSE Score', 'Age', 'Hippocampal Volume', 'APOE Status', 
               'Memory Test', 'Education Level', 'CDR Score', 'ADL Score',
               'Diabetes', 'Hypertension', 'BMI', 'Physical Activity']
    
    importance = [0.185, 0.165, 0.142, 0.128, 0.095, 0.082, 0.075, 0.063, 0.045, 0.038, 0.035, 0.027]
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    
    # Create horizontal bar plot
    bars = plt.barh(sorted_features, sorted_importance, color=plt.cm.viridis(np.linspace(0, 1, len(features))))
    
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title('Clinical Feature Importance for Alzheimer Detection', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, imp in zip(bars, sorted_importance):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{imp:.3f}', ha='left', va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'feature_importance.png'

def create_all_plots():
    """Create all plots and return their filenames"""
    print("📊 Creating comprehensive analysis plots...")
    
    plots = {}
    
    try:
        plots['accuracy_energy'] = create_model_accuracy_plot()
        print("✅ Created Model Accuracy vs Energy plot")
    except Exception as e:
        print(f"❌ Error creating accuracy plot: {e}")
    
    try:
        plots['power_output'] = create_power_output_plot()
        print("✅ Created Power Output comparison plot")
    except Exception as e:
        print(f"❌ Error creating power plot: {e}")
    
    try:
        plots['performance_comparison'] = create_model_performance_comparison()
        print("✅ Created Model Performance comparison plot")
    except Exception as e:
        print(f"❌ Error creating performance plot: {e}")
    
    try:
        plots['confusion_matrix'] = create_confusion_matrix_plot()
        print("✅ Created Confusion Matrix plot")
    except Exception as e:
        print(f"❌ Error creating confusion matrix: {e}")
    
    try:
        plots['roc_curves'] = create_roc_curves()
        print("✅ Created ROC Curves plot")
    except Exception as e:
        print(f"❌ Error creating ROC curves: {e}")
    
    try:
        plots['training_history'] = create_training_history()
        print("✅ Created Training History plot")
    except Exception as e:
        print(f"❌ Error creating training history: {e}")
    
    try:
        plots['feature_importance'] = create_feature_importance_plot()
        print("✅ Created Feature Importance plot")
    except Exception as e:
        print(f"❌ Error creating feature importance: {e}")
    
    print(f"🎉 All plots saved in: {RESULTS_DIR}")
    return plots

# Add this function to your Flask app to display plots
def add_plots_route(app):
    """Add a route to display all generated plots"""
    @app.route('/results')
    def show_results():
        if 'user' not in session:
            return redirect(url_for('login'))
        
        # Generate all plots
        plot_files = create_all_plots()
        
        return render_template('results.html', 
                             plots=plot_files,
                             results_dir=RESULTS_DIR,
                             user=session.get('user'))

# Add this to your main Flask app after creating the app
# Call this function to create all plots when needed
if __name__ == '__main__':
    # Create all plots
    plot_files = create_all_plots()
    print("📈 Generated plots:")
    for plot_name, filename in plot_files.items():
        print(f"   📊 {plot_name}: {filename}")