# Sydney Rain Prediction — End-to-End Machine Learning Project

[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Garv_Sachdeva-blue?logo=linkedin)](https://www.linkedin.com/in/garvsachdeva)

---

## 📌 Project Overview

This project predicts whether it will rain tomorrow in Sydney using historical weather data. It is a comprehensive end-to-end solution that includes:

- Advanced data preprocessing and feature engineering  
- Handling missing data with KNN imputation  
- Outlier detection and clipping for key meteorological features  
- Scaling and encoding of features  
- Multiple model training: Logistic Regression, Random Forest, XGBoost, LightGBM  
- Hyperparameter tuning via randomized search with cross-validation  
- Stacking ensemble to improve prediction robustness  
- Explainability using SHAP values to interpret model predictions  
- Evaluation with multiple metrics: Accuracy, F1-score, ROC-AUC  
- Visualization of results including ROC curves, feature importances, and SHAP summary plots  
- Saved trained models and scalers for deployment-ready use

---

## 🎯 Motivation

Predicting rainfall accurately is critical for agriculture, disaster preparedness, and urban planning. This project demonstrates mastery over the complete machine learning lifecycle, showcasing skills that align with top product-based companies’ expectations for data scientists and ML engineers.

---

## 📂 Dataset

- **Source:** [Australian Bureau of Meteorology - Sydney Weather Dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)  
- **Size:** 3337 samples, 17 meteorological features  
- **Target:** `RainTomorrow` (binary classification — yes/no)

---

## 🛠️ Key Features & Engineering

- Converted dates to datetime and extracted useful components  
- Imputed missing values using KNN Imputer for robust handling of sparse data  
- Clipped outliers in Rainfall, Evaporation, and Sunshine using IQR method  
- Created new feature `WindDiff` = `WindGustSpeed` - `WindSpeed9am` to capture wind variation  
- Encoded categorical variables (`WindGustDir`, `WindDir9am`, `WindDir3pm`) with one-hot encoding  
- Scaled numerical features using StandardScaler to normalize distributions

---

## 🤖 Models and Techniques

| Model                | Accuracy | F1-Score | ROC-AUC |
|----------------------|----------|----------|---------|
| Logistic Regression  | 82.78%   | 0.61     | 0.84    |
| Random Forest        | 83.38%   | 0.62     | 0.85    |
| XGBoost              | 81.13%   | 0.59     | 0.85    |
| LightGBM             | 83.08%   | 0.63     | 0.86    |
| **Stacking Ensemble**| **83.68%**| **0.63** | **0.86**|

- Tuned XGBoost using RandomizedSearchCV, improving ROC-AUC to ~0.865  
- Built a stacking ensemble using top-performing models  
- Applied Stratified K-Fold cross-validation for evaluation stability

---

## 📊 Model Explainability

- Used **SHAP** (SHapley Additive exPlanations) to interpret feature contributions  
- SHAP summary plots highlight the most impactful features on rain prediction  
- Ensures model transparency and trust for end-users and stakeholders

---

## 📈 Visualizations

- ROC Curves comparing model performances  
- Feature Importance plots from tree-based models  
- SHAP Summary Plots for explainability  

_All plots are saved under `/outputs/plots/`_

---

## 💻 Installation

```bash
git clone https://github.com/GARVS0205/sydney-rain-prediction.git
cd sydney-rain-prediction
pip install -r requirements.txt
```

## 🚀 Usage
To run the project locally:
```bash
python main.py
```
This will:
- Preprocess the data
- Train and evaluate all models
- Generate and save all visualizations
- Save final trained models and scaler for deployment

## 📄 License
This project is licensed under the MIT License.
See the LICENSE file for more details.

## 📞 Contact
If you have any questions or would like to collaborate:
[GitHub: GARVS0205](https://github.com/GARVS0205)
[LinkedIn: Garv Sachdeva](https://www.linkedin.com/in/garv-sachdeva-758676269)
[Email: garvsachdeva02@gmail.com](mailto:garvsachdeva02@gmail.com)


🙏 Thank You!
Thanks for checking out my project!
If you found it helpful or impressive, feel free to ⭐️ star the repo — it really helps and means a lot 😊


