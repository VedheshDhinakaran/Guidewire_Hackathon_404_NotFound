
# 📘 Kubernetes Cluster Failure Prediction Project

## 🧠 Project Title:
**Kubernetes Failure  Prediction in Guidewire Cloud Environments using Random Forest**

---

## 📌 Project Overview

This machine learning project aims to accurately classify different types of failure scenarios in a Guidewire-based cloud deployment (Kubernetes environment). Using a supervised learning approach with a Random Forest classifier, the model analyzes various system metrics to predict failure types — a crucial step in automated monitoring and predictive maintenance in production environments.

---

## 🗂️ Dataset Description

- **File Used**: `k8s_failures_updated.csv`
- **Total Records**: ~3800+ entries (assumed from standard logs)
- **Target Feature**: `failure_type`
- **Removed Column**: `Timestamp` (not relevant for classification)
- **Independent Variables**: A wide set of performance and usage metrics like CPU utilization, memory, container lifecycle metrics, etc.

---

## 🛠 Technologies and Libraries

- **Language**: Python 3.x
- **Libraries Used**:
  - `pandas`, `numpy` – Data manipulation
  - `scikit-learn` – ML model, preprocessing, evaluation
  - `matplotlib`, `seaborn` – Data visualization

---

## 🔍 How It Works

### 1. **Library Installation**
```python
!pip install pandas scikit-learn matplotlib seaborn
```

### 2. **Importing Required Modules**
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

### 3. **Loading the Dataset**
```python
df = pd.read_csv('path/to/k8s_failures_updated.csv')
df.drop(columns=['Timestamp'], inplace=True)
```

### 4. **Data Preprocessing**
- Dropping non-useful columns (`Timestamp`)
- Label Encoding for the target variable
- Standardization using `StandardScaler`

```python
X = df.drop(columns=['failure_type'])
y = df['failure_type']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. **Splitting Dataset**
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
```

### 6. **Model Training**
```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

### 7. **Model Evaluation**
```python
y_pred = rf_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```

### 8. **Feature Importance Visualization**
```python
importances = rf_model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title('Feature Importance for Failure Classification')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
```

---

## 📈 Output Summary

- **Accuracy**: ~94%+ depending on test split and feature variety
- **Metrics**: Precision, Recall, F1-score across failure types
- **Confusion Matrix**: For analyzing misclassifications
- **Feature Importance Chart**: To interpret model behavior and critical features

---

## 💡 Key Takeaways

- Random Forests offer high performance with minimal tuning.
- Preprocessing and encoding have a direct impact on classification quality.
- Feature importance helps with root cause identification in system failures.

---

## 🧭 Future Work

- Hyperparameter tuning with GridSearchCV
- Trying more complex models like XGBoost, LightGBM
- Deploying the model in a monitoring pipeline using Flask or FastAPI
- Real-time alerting and visualization dashboards (e.g., Grafana)

---

## 👩‍💻 Author

**NehaaV, VedheshDhinakaran, DhimantKulkarni, HimaSagar**  
3rd Semester | CSE  
Kubernetes Cluster Failure Prediction

