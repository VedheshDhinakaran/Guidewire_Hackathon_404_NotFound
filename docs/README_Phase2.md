
# K8s Failure Prediction and Remediation using Random Forest

## 📌 Project Overview

This project focuses on identifying and classifying failure types in Kubernetes (K8s) environments and providing solution using machine learning techniques. A Random Forest classifier is employed to process failure-related metrics and accurately predict the type of failure. The objective is to enable proactive mitigation strategies in real-time Kubernetes systems.

## 📂 Dataset

- **Source**: `k8s_failures_updated.csv`
- **Target Column**: `failure_type`
- **Features**: Various numerical attributes associated with system performance and states.
- **Preprocessing**: The `Timestamp` column, which does not contribute to classification, is dropped.

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Make sure Python 3.x is installed with the necessary libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Update the dataset path in the notebook:
   ```python
   file_path = 'C:/Users/NEHAA.V/Desktop/k8s_failures_updated.csv'
   ```
4. Run all cells in the notebook to see preprocessing, training, evaluation, and visualization results.

## 🧠 Code Breakdown

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Load and Clean Dataset
```python
df = pd.read_csv(file_path)
df.drop(columns=['Timestamp'], inplace=True)
```

### 3. Feature and Target Separation
```python
X = df.drop(columns=['failure_type'])
y = df['failure_type']
```

### 4. Label Encoding and Scaling
```python
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
```

### 6. Model Training
```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

### 7. Model Evaluation
```python
y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### 8. Feature Importance Visualization
```python
importances = rf_model.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance')
plt.show()
```

## 📊 Expected Output

- **Classification Report**: Insight into precision, recall, and F1-score per failure class.
- **Accuracy Score**: Overall percentage of correctly predicted labels.
- **Confusion Matrix**: Evaluates the performance in identifying different types of failures.
- **Feature Importance Chart**: Highlights key system attributes impacting model predictions.

## 📈 Future Improvements

- Hyperparameter tuning of the Random Forest model.
- Try advanced models like XGBoost, SVM, or deep learning approaches.
- Deploy as a REST API for real-time inference.

## 🧑‍💻 Author

**NehaaV, VedheshDhinakaran, DhimantKulkarni, HimaSagar** 
3rd Semester CSE  
Project on AI-based solution for Kubernetes Cluster Management.
