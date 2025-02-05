import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve

# Load dataset
df = pd.read_csv("customer_churn.csv")  # Ensure correct file path

# Display basic info
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical with mode
    else:
        df[col].fillna(df[col].median(), inplace=True)   # Fill numerical with median

# Verify missing values are handled
print(df.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Subscription Type', 'Contract Length']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature scaling
scaler = MinMaxScaler()
scaling_columns = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                   'Payment Delay', 'Total Spend', 'Last Interaction']
df[scaling_columns] = scaler.fit_transform(df[scaling_columns])

# Define features and target
X = df.drop(columns=['CustomerID', 'Churn'])  # Drop ID and target
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Visualizing Class distribution
sns.countplot(x='Churn', data=df)
plt.show()

#Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

#Random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

#Model evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred):.2f}")

print("Logistic Regression:")
evaluate_model(log_model, X_test, y_test)

print("Random Forest:")
evaluate_model(rf_model, X_test, y_test)

print("XGBoost:")
evaluate_model(xgb_model, X_test, y_test)

#Data Visualization
# Feature importance(Random forest)
importances = rf_model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

#ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label="Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
