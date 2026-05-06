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
The dataset is highly imbalanced:
- ~88% No Stroke (0)
- ~12% Stroke (1)

Accuracy alone is a misleading metric. Recall Macro is more meaningful 
for model comparison, as it treats both classes equally and is consistent 
with the GridSearchCV scoring metric used during tuning.

---

## Project Pipeline

### 1. Data Preparation
- Set `id` as the DataFrame index
- Handled missing values in `bmi` using `SimpleImputer` (median strategy)
- Encoded categorical features using `OneHotEncoder`
- Scaled numerical features using `StandardScaler`
- Used `ColumnTransformer` to build a clean preprocessing pipeline

### 2. Modeling
Three classification algorithms were tested, each evaluated with default 
parameters and then tuned using `GridSearchCV` with `scoring='recall_macro'`
and `cv=3`.

| Model | Strategy |
|---|---|
| Decision Tree | Default only (baseline) |
| Logistic Regression | Default + GridSearchCV tuning |
| K-Nearest Neighbors | Default + GridSearchCV tuning |
| Random Forest | Default + GridSearchCV tuning |

### 3. Hyperparameter Tuning
**Logistic Regression:**
- Tuned: solver, penalty (l1/l2/elasticnet/None), C values, class_weight
- Best: solver=saga, penalty=l1, C=0.1, class_weight=balanced

**KNN:**
- Tuned: n_neighbors, metric
- Best: metric=manhattan, n_neighbors=3

**Random Forest:**
- Tuned: n_estimators, max_depth, min_samples_split, class_weight
- Best: class_weight=balanced, max_depth=5, min_samples_split=10, n_estimators=200

---

## Results

### Baseline - Decision Tree (Default)
| Metric | Training Set | Test Set |
|---|---|---|
| Accuracy | 100% | 86% |
| Precision (Stroke) | 100% | 38% |
| Recall (Stroke) | 100% | 32% |

### Model Comparison (Test Set - Tuned Models)
| Model | Recall Macro | Recall (Stroke) | Accuracy |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.75** | **0.76** | 0.74 |
| Random Forest | 0.69 | 0.59 | 0.76 |
| KNN | 0.52 | 0.09 | 0.85 |

---

## Recommended Model: Tuned Logistic Regression

The Tuned Logistic Regression achieved the highest Recall Macro (0.75),
correctly identifying 76% of real stroke patients.

In a medical context, **Recall is the most critical metric**, a missed 
stroke can lead to death or severe disability, while a false alarm only 
results in further testing.

---

## Key Observations
- Decision Tree showed clear **overfitting** (100% train vs 86% test accuracy)
- KNN had high accuracy (85%) but nearly **failed to detect strokes** (recall=0.09)
- Using `class_weight='balanced'` significantly improved stroke detection
- **Accuracy is misleading** with imbalanced data, Recall Macro is the 
  appropriate metric for this problem
