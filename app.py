"""
Heart Disease Classification - Interactive Streamlit App
BITS WILP - Machine Learning Assignment 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">❤️ Heart Disease Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Models for Predicting Heart Disease</p>', unsafe_allow_html=True)

# Model information
MODEL_INFO = {
    'Logistic Regression': {
        'file': 'models/logistic_regression.pkl',
        'description': 'A linear model for binary classification using logistic function'
    },
    'Decision Tree': {
        'file': 'models/decision_tree.pkl',
        'description': 'A tree-based model that makes decisions using feature splits'
    },
    'K-Nearest Neighbors': {
        'file': 'models/knn.pkl',
        'description': 'Classifies based on the majority class of k nearest neighbors'
    },
    'Naive Bayes': {
        'file': 'models/naive_bayes.pkl',
        'description': 'Probabilistic classifier based on Bayes theorem (Gaussian)'
    },
    'Random Forest': {
        'file': 'models/random_forest.pkl',
        'description': 'Ensemble of decision trees for robust predictions'
    },
    'XGBoost': {
        'file': 'models/xgboost.pkl',
        'description': 'Gradient boosting ensemble for high-performance classification'
    }
}

# Feature names for the heart disease dataset
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Load models and scaler
@st.cache_resource
def load_model(model_path):
    """Load a trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load the trained scaler"""
    try:
        with open('models/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

def validate_dataset(df):
    """Validate uploaded dataset"""
    # Check if target column exists
    if 'target' not in df.columns:
        st.error("❌ Dataset must contain a 'target' column")
        return False
    
    # Check if all required features are present
    required_features = set(FEATURE_NAMES)
    dataset_features = set(df.columns) - {'target'}
    
    if not required_features.issubset(dataset_features):
        missing = required_features - dataset_features
        st.error(f"❌ Missing features: {missing}")
        return False
    
    # Check for missing values
    if df.isnull().any().any():
        st.warning("⚠️ Dataset contains missing values. They will be dropped.")
        
    return True

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC Score'] = 0.0
    else:
        metrics['AUC Score'] = 0.0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    return fig

def plot_metrics_comparison(metrics):
    """Create bar plot of metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    colors = ['#FF4B4B', '#FFA500', '#4CAF50', '#2196F3', '#9C27B0', '#FF69B4']
    bars = ax.bar(metric_names, metric_values, color=colors[:len(metric_names)], alpha=0.7)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

# Sidebar
st.sidebar.header("📊 Configuration")

# Model selection
st.sidebar.subheader("🤖 Select Model")
selected_model = st.sidebar.selectbox(
    "Choose a classification model:",
    list(MODEL_INFO.keys()),
    help="Select which ML model to use for predictions"
)

st.sidebar.markdown(f"**Description:** {MODEL_INFO[selected_model]['description']}")

# File upload section
st.sidebar.subheader("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with test data",
    type=['csv'],
    help="Upload a CSV file containing test data with the same features as training data"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📋 Required Features:
- age, sex, cp, trestbps, chol
- fbs, restecg, thalach, exang
- oldpeak, slope, ca, thal
- **target** (0 = No Disease, 1 = Disease)
""")

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
        
        # Display dataset preview
        with st.expander("👀 Preview Dataset", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", df.shape[0])
            with col2:
                st.metric("Features", df.shape[1] - 1)
            with col3:
                missing = df.isnull().sum().sum()
                st.metric("Missing Values", missing)
        
        # Validate dataset
        if validate_dataset(df):
            # Drop missing values
            df = df.dropna()
            
            # Separate features and target
            X_test = df[FEATURE_NAMES]
            y_test = df['target']
            
            st.markdown("---")
            
            # Load model and scaler
            model = load_model(MODEL_INFO[selected_model]['file'])
            scaler = load_scaler()
            
            if model is not None and scaler is not None:
                # Scale features
                X_test_scaled = scaler.transform(X_test)
                
                # Make predictions
                with st.spinner(f'🔄 Making predictions using {selected_model}...'):
                    y_pred = model.predict(X_test_scaled)
                    
                    # Get probability predictions if available
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_pred_proba = None
                
                st.success(f"✅ Predictions completed using {selected_model}!")
                
                # Calculate metrics
                metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Display metrics
                st.markdown("## 📈 Model Performance Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("📊 Precision", f"{metrics['Precision']:.4f}")
                
                with col2:
                    st.metric("🔄 Recall", f"{metrics['Recall']:.4f}")
                    st.metric("⚖️ F1 Score", f"{metrics['F1 Score']:.4f}")
                
                with col3:
                    st.metric("📉 AUC Score", f"{metrics['AUC Score']:.4f}")
                    st.metric("🔗 MCC Score", f"{metrics['MCC']:.4f}")
                
                st.markdown("---")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎨 Confusion Matrix")
                    fig_cm = plot_confusion_matrix(y_test, y_pred)
                    st.pyplot(fig_cm)
                    plt.close()
                
                with col2:
                    st.markdown("### 📊 Metrics Comparison")
                    fig_metrics = plot_metrics_comparison(metrics)
                    st.pyplot(fig_metrics)
                    plt.close()
                
                # Classification report
                st.markdown("---")
                st.markdown("### 📋 Detailed Classification Report")
                
                target_names = ['No Disease (0)', 'Disease (1)']
                report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                
                # Prediction distribution
                st.markdown("---")
                st.markdown("### 📊 Prediction Distribution")
                
                col1, col2, col3 = st.columns(3)
                
                pred_counts = pd.Series(y_pred).value_counts()
                true_counts = pd.Series(y_test).value_counts()
                
                with col1:
                    st.metric("Total Predictions", len(y_pred))
                
                with col2:
                    no_disease = pred_counts.get(0, 0)
                    st.metric("Predicted: No Disease", no_disease)
                
                with col3:
                    disease = pred_counts.get(1, 0)
                    st.metric("Predicted: Disease", disease)
                
            else:
                st.error("❌ Failed to load model or scaler. Please check if model files exist.")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to begin")
    
    st.markdown("## 🚀 How to Use This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📝 Steps:
        1. **Select a Model** from the sidebar dropdown
        2. **Upload CSV File** with test data
        3. **View Predictions** and performance metrics
        4. **Analyze Results** using visualizations
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Available Models:
        - Logistic Regression
        - Decision Tree Classifier
        - K-Nearest Neighbors
        - Naive Bayes (Gaussian)
        - Random Forest (Ensemble)
        - XGBoost (Ensemble)
        """)
    
    st.markdown("---")
    
    st.markdown("## 📚 About the Dataset")
    st.markdown("""
    This application uses the **Heart Disease Dataset** from Kaggle.
    
    **Key Features:**
    - **13 Features**: Including age, sex, chest pain type, blood pressure, cholesterol, etc.
    - **Target**: Binary classification (0 = No Disease, 1 = Disease)
    - **Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
    
    **Sample CSV Format:**
    ```
    age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
    37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Heart Disease Classification ML App</strong></p>
    <p>BITS WILP | Machine Learning Assignment 2 | 2026</p>
</div>
""", unsafe_allow_html=True)
