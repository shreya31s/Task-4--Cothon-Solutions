# Churn Prediction Model for Subscription Services

## Project Overview

The Churn Prediction Model aims to predict whether a customer will cancel their subscription based on historical user activity, demographics, and usage patterns. This project utilizes machine learning classification techniques to improve customer retention strategies by identifying at-risk customers.

## Features
- Predicts customer churn using historical data.
- Utilizes Logistic Regression, Random Forest, and XGBoost for classification.
- Includes feature engineering for user engagement and subscription length.
- Implements data visualization to understand churn trends.
- Evaluates model performance using Accuracy, Precision, Recall, and AUC-ROC.
- Ensures scalability and efficiency in large datasets.

## 🔧 Tech Stack

- Programming Languages & Libraries
- Python 🐍
- Pandas, NumPy - Data manipulation & analysis
- Scikit-learn - Machine Learning models
- Seaborn, Matplotlib - Data visualization
- XGBoost - Advanced boosting techniques
- Statsmodels - Statistical modeling

## Project Structure
```Churn Prediction/
│── data/                      # Dataset storage
│   ├── churn_data.csv         # Raw dataset
│── models/                    # Trained models
│   ├── churn_model.pkl        # Saved model
│── src/                       # Source code
│   ├── churn_prediction.py    # Main script
│   ├── data_preprocessing.py  # Data cleaning & feature engineering
│   ├── model_training.py      # Model training & evaluation
│── notebooks/                 # Jupyter Notebooks (Exploratory Data Analysis)
│── README.md                  # Project documentation
│── requirements.txt           # Dependencies
│── .gitignore                 # Git ignore file
```
## 📊 Model Performance

- Accuracy: 85%
- Precision: 82%
- Recall: 78%
- AUC-ROC Score: 0.88

## 📌 Results & Insights

- Customers with higher support calls and payment delays are more likely to churn.
- Monthly subscribers tend to churn more than yearly subscribers.
- Tenure (longer subscription period) reduces churn probability.
- The model helps businesses identify high-risk customers and take proactive retention actions.

### Contact
- Name: Shreya. S
- Email: shreyasuresh3107@gmail.com
- linkdn: www.linkedin.com/in/shreya-suresh-620922256
