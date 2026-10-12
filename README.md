# 🚀 Software Project Risk Prediction System

An AI/ML powered system to predict software project risks (High/Medium/Low) using Random Forest & SHAP Explainability.

**Author:** Pranali Ghadage  
**Repository:** https://github.com/ghadagepranali/software-risk-predictor

### 🎯 Features
- Predicts Project Risk based on Progress, Budget, Team Size, Bugs
- SHAP based Explainability - Why risk is High?
- Visual Analytics - Risk Distribution
- Streamlit Web App

### 📁 Project Files
- `app.py` - Main Streamlit Application
- `model.py` - ML Model (Random Forest)
- `create_dataset.py` - Generates Synthetic Dataset
- `analyze_data.py` - EDA & Visualizations
- `requirements.txt` - Dependencies

### ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
