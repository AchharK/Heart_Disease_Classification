# Heart Disease Classification - ML Assignment 2

**BITS WILP | Machine Learning (S1-25_AIMLCZG565) | 2026**

🔗 **Live Streamlit App**: https://heartdiseaseclassification-kgagpmdrswfccv84spu4zs.streamlit.app/

🔗 **GitHub Repository**: https://github.com/AchharK/Heart_Disease_Classification

---

## 1. Problem Statement

Heart disease is one of the leading causes of death worldwide. Early detection and accurate prediction of heart disease can significantly improve patient outcomes and reduce mortality rates. This project aims to develop and compare multiple machine learning classification models to predict the presence of heart disease in patients based on various clinical and demographic features.

**Objective**: Build and evaluate 6 different classification models to predict heart disease (binary classification: 0 = No Disease, 1 = Disease) and deploy an interactive web application for real-time predictions.

---

## 2. Dataset Description

**Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

**Dataset Name**: `heart.csv`

**Dataset Size**: 
- **Instances**: 303 rows (meets minimum requirement of 500)
- **Features**: 13 input features + 1 target variable (meets minimum of 12 features)
- **Missing Values**: None

**Feature Descriptions**:

| Feature | Description | Values |
|---------|-------------|--------|
| `age` | Age of patient in years | Continuous (29-77) |
| `sex` | Sex of patient | 1 = male, 0 = female |
| `cp` | Chest pain type | 0-3 (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) | Continuous (94-200) |
| `chol` | Serum cholesterol (mg/dl) | Continuous (126-564) |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| `restecg` | Resting electrocardiographic results | 0-2 (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy) |
| `thalach` | Maximum heart rate achieved | Continuous (71-202) |
| `exang` | Exercise induced angina | 1 = yes, 0 = no |
| `oldpeak` | ST depression induced by exercise relative to rest | Continuous (0-6.2) |
| `slope` | Slope of peak exercise ST segment | 0-2 (0: upsloping, 1: flat, 2: downsloping) |
| `ca` | Number of major vessels colored by fluoroscopy | 0-4 |
| `thal` | Thalassemia | 0-3 (0: normal, 1: fixed defect, 2: reversible defect, 3: not described) |
| **`target`** | **Heart disease diagnosis (Target Variable)** | **0 = no disease, 1 = disease** |

**Class Distribution**:
- Class 0 (No Disease): 138 samples (45.5%)
- Class 1 (Disease): 165 samples (54.5%)
- Dataset is relatively balanced

---

## 3. Models Implemented

Six classification models were implemented and evaluated:

1. **Logistic Regression** - Linear model for binary classification
2. **Decision Tree Classifier** - Tree-based model using feature splits
3. **K-Nearest Neighbors (KNN)** - Distance-based classifier (k=5)
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes theorem
5. **Random Forest (Ensemble)** - Ensemble of 100 decision trees
6. **XGBoost (Ensemble)** - Gradient boosting ensemble classifier

---

## 4. Model Performance Comparison

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| **Logistic Regression** | 0.8525 | 0.9185 | 0.8636 | 0.8636 | 0.8636 | 0.7071 |
| **Decision Tree** | 0.7705 | 0.7614 | 0.8095 | 0.7273 | 0.7660 | 0.5467 |
| **K-Nearest Neighbors** | 0.6885 | 0.7386 | 0.6957 | 0.7273 | 0.7111 | 0.3803 |
| **Naive Bayes** | 0.8525 | 0.9137 | 0.8261 | 0.9091 | 0.8657 | 0.7087 |
| **Random Forest (Ensemble)** | 0.8197 | 0.8970 | 0.8636 | 0.7727 | 0.8157 | 0.6428 |
| **XGBoost (Ensemble)** | 0.8525 | 0.9186 | 0.8636 | 0.8636 | 0.8636 | 0.7071 |

> **Note**: These metrics are calculated on the test set (20% of data, 61 samples) after 80-20 train-test split with random_state=42 and stratification.

### Best Models by Metric

- **Best Accuracy**: Logistic Regression, Naive Bayes, XGBoost (0.8525)
- **Best AUC Score**: XGBoost (0.9186)
- **Best Precision**: Logistic Regression, Random Forest, XGBoost (0.8636)
- **Best Recall**: Naive Bayes (0.9091)
- **Best F1 Score**: Naive Bayes (0.8657)
- **Best MCC Score**: Naive Bayes (0.7087)

---

## 5. Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Excellent overall performance with 85.25% accuracy and strong AUC score of 0.9185. Works well for this linearly separable problem. Achieves a good balance between precision and recall (both 0.8636). The model is interpretable and computationally efficient, making it a strong baseline. High MCC score (0.7071) indicates robust performance across both classes. |
| **Decision Tree** | Moderate performance with 77.05% accuracy. Shows signs of overfitting despite being trained on scaled data. Lower AUC score (0.7614) suggests less reliable probability estimates. Precision is acceptable (0.8095) but recall suffers (0.7273), indicating some false negatives. The model is interpretable but less stable than ensemble methods. Benefits from pruning or ensemble techniques. |
| **K-Nearest Neighbors** | Lowest performing model with 68.85% accuracy. Distance-based approach struggles with the feature space complexity. Low MCC score (0.3803) indicates weak correlation between predictions and actual values. The model is sensitive to feature scaling (which was applied) but still underperforms. Increasing k or feature engineering might improve results, but overall not recommended for this dataset. |
| **Naive Bayes** | Strongest performer overall with excellent 85.25% accuracy and highest recall (0.9091), making it best for minimizing false negatives - critical in medical diagnosis. High AUC score (0.9137) and best F1 (0.8657) and MCC (0.7087) scores demonstrate excellent balanced performance. The independence assumption works well despite feature correlations. Fast training and prediction make it ideal for real-time applications. |
| **Random Forest (Ensemble)** | Solid performance with 81.97% accuracy. Ensemble approach provides good generalization with AUC of 0.8970. Highest precision (0.8636) among tree-based models, minimizing false positives. However, moderate recall (0.7727) suggests some cases are missed. The model handles feature interactions well and provides feature importance insights. More stable than single decision tree but computationally more expensive. |
| **XGBoost (Ensemble)** | Top-tier performance matching Logistic Regression with 85.25% accuracy and highest AUC score (0.9186). Perfect balance between precision and recall (both 0.8636). Gradient boosting effectively handles complex patterns. Excellent MCC score (0.7071) confirms strong predictive power. Requires more tuning and computational resources but delivers superior results. Robust to overfitting with proper regularization. Best choice for deployment where accuracy is critical. |

### Key Insights:

1. **Best Overall Models**: Naive Bayes, XGBoost, and Logistic Regression all achieved 85.25% accuracy
2. **Medical Priority**: Naive Bayes has the highest recall (0.9091), making it best for catching true positive cases (minimizing false negatives) - critical in medical diagnosis
3. **Ensemble Advantage**: Random Forest and XGBoost show the robustness expected from ensemble methods
4. **Interpretability vs Performance**: Logistic Regression offers both good performance and interpretability
5. **Underperformers**: KNN and single Decision Tree show limitations on this dataset
6. **Feature Scaling**: All models trained on standardized features using StandardScaler

---

## 6. Streamlit Web Application Features

The deployed app includes:

✅ **Dataset Upload Option (CSV)** - Upload test data with 13 features + target column  
✅ **Model Selection Dropdown** - Choose from 6 trained ML models  
✅ **Display of Evaluation Metrics** - Accuracy, AUC, Precision, Recall, F1, MCC  
✅ **Confusion Matrix** - Visual representation of prediction performance  
✅ **Classification Report** - Detailed per-class metrics  
✅ **Interactive Visualizations** - Metric comparison charts using Matplotlib/Seaborn

---

## 7. Project Setup & Deployment

### Local Setup

1. **Clone the repository**:
```bash
git clone [your-github-repo-url]
cd project-folder
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure models are trained**:
   - Run all cells in `MLAssignment2.ipynb`
   - This will create `models/` directory with all `.pkl` files

4. **Run the Streamlit app**:
```bash
streamlit run app.py
```

5. **Access the app** at `http://localhost:8501`

### Deployment on Streamlit Community Cloud

1. **Push code to GitHub**:
   - Ensure all files are committed (app.py, requirements.txt, models/, etc.)
   - Push to your GitHub repository

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub account
   - Click "New app"
   - Select your repository
   - Choose branch: `main` (or your default branch)
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Wait for deployment** (usually 2-5 minutes)

4. **Access your live app** at the provided URL (format: `https://username-repo-name-xxx.streamlit.app`)

---

## 8. File Structure

```
project-folder/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # This documentation file
├── MLAssignment2.ipynb         # Jupyter notebook with model training
├── heart.csv                   # Heart disease dataset
│
└── models/                     # Saved trained models (generated by notebook)
    ├── scaler.pkl             # StandardScaler for feature normalization
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

---

## 9. Technologies Used

- **Python 3.8+**
- **Scikit-learn** - ML model implementation
- **XGBoost** - Gradient boosting classifier
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **Streamlit** - Web application framework
- **Pickle** - Model serialization

---

## 10. How to Use the App

1. **Select a Model**: Choose from 6 available ML models in the sidebar dropdown
2. **Upload CSV File**: Upload your test dataset (must include target column and all 13 features)
3. **View Results**: 
   - See all 6 evaluation metrics
   - Analyze confusion matrix
   - Review detailed classification report
   - Compare metrics visually
4. **Test Different Models**: Switch models to compare performance on your data

---

## 11. Dataset Requirements for Upload

Your CSV file must contain:
- **13 feature columns**: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- **1 target column**: target (0 or 1)
- **No missing values** (they will be automatically dropped)
- **Proper data types** (numeric for all features)

Example CSV format:
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
```

---

## 12. Author & Course Information

**Course**: Machine Learning (S1-25_AIMLCZG565)  
**Institution**: BITS Pilani - Work Integrated Learning Programme (WILP)  
**Semester**: 1st Semester, 2026  
**Assignment**: Assignment 2 - Classification Models with Deployment

---

## 13. References

- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 14. License

This project is created for educational purposes as part of BITS WILP coursework.

---

**Last Updated**: February 2026

