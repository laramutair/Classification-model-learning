# Classification-model-learning
# Stroke Prediction – Classification Model

A machine learning project that builds a classification model to predict the likelihood of a stroke based on patient health data.

---

## Dataset Overview

| Property | Details |
|----------|---------|
| **DS** | Stroke Prediction Dataset |
| **Rows** | 1,137 patients |
| **Features** | 11 input features + 1 target |
| **Target** | `stroke` (0 = No Stroke, 1 = Stroke) |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `id` | int | Unique patient identifier (used as index) |
| `gender` | categorical | Male / Female / Other |
| `age` | numerical | Patient age |
| `hypertension` | binary | 0 = No, 1 = Yes |
| `heart_disease` | binary | 0 = No, 1 = Yes |
| `ever_married` | categorical | Yes / No |
| `work_type` | categorical | Private, Govt_job, Self-employed, children, Never_worked |
| `Residence_type` | categorical | Urban / Rural |
| `avg_glucose_level` | numerical | Average blood glucose level |
| `bmi` | numerical | Body Mass Index |
| `smoking_status` | categorical | never smoked, formerly smoked, smokes, Unknown |

---

## Class Imbalance

The dataset is highly **imbalanced**:
- **~88%** No Stroke (0)
- **~12%** Stroke (1)

This means accuracy alone is a misleading metric, the model can score high by simply predicting the majority class. Metrics like **Recall** and **Precision** are more meaningful here.

---

## Project Pipeline

### 1. Data Preparation
- Set `id` as the DataFrame index (not dropped, preserved for reference)
- Checked and handled missing values in `bmi` using `SimpleImputer`
- Encoded categorical features using `OneHotEncoder`
- Scaled numerical features using `StandardScaler`
- Used `ColumnTransformer` to build a clean preprocessing pipeline

### 2. Modeling
- Model used: **Decision Tree Classifier** (default parameters)
- Split: Train / Test using `train_test_split`
- Pipeline: `Preprocessor → DecisionTreeClassifier`

### 3. Evaluation
The model was evaluated using:
- Accuracy (Train & Test)
- Confusion Matrix (Raw Counts + Normalized)
- Classification Report (Precision, Recall, F1-Score)

---

## Results

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **Accuracy** | 100% | 86% |
| **Precision** (Stroke) |100% | 38% |
| **Recall** (Stroke) | 100% | 32% |

### Key Observations

- The **100% training accuracy** vs **86% test accuracy** indicates clear **overfitting**
- The **86% test accuracy is misleading** due to class imbalance, the model can achieve high accuracy by predicting "No Stroke" most of the time
- **Precision = 38%**: When the model predicted a stroke, it was correct only 38% of the time
- **Recall = 32%**: The model correctly identified only 32% of actual stroke patients, missing 68% of real cases

---

## Most Important Metric: Recall

> In a medical context, **Recall** is the most critical metric.

A **False Negative** (missing a real stroke patient) is far more dangerous than a **False Positive**. A missed stroke can lead to death, while a false alarm only results in further testing.

Therefore, optimizing for **Recall** ensures we catch as many actual stroke cases as possible.

---
