# Project 2: Fraud Detection Pipeline using Machine Learning

## Overview

This project focuses on building a fraud detection system using supervised machine learning techniques. The objective is to identify fraudulent credit card transactions from a highly imbalanced dataset and evaluate model performance using appropriate classification metrics.

The project demonstrates the complete machine learning workflow, including data preparation, class imbalance handling using SMOTE, model training, and performance evaluation.

---

## Objectives

* Build a fraud detection classification model.
* Handle severe class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
* Train and compare multiple machine learning algorithms.
* Evaluate models using Precision, Recall, and ROC-AUC instead of Accuracy.
* Select the best-performing model for fraud detection.

---

## Dataset

**Dataset:** Credit Card Fraud Detection Dataset

### Features

The dataset contains:

* Time
* Amount
* V1 to V28 (anonymized features generated through PCA)
* Class (Target Variable)

### Target Variable

| Class | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

### Dataset Statistics

* Total Transactions: 284,807
* Legitimate Transactions: 284,315
* Fraudulent Transactions: 492

This extreme imbalance makes fraud detection a challenging classification problem.

---

## Data Preprocessing

### Train-Test Split

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

Stratified sampling was used to preserve class distribution.

---

## Handling Class Imbalance

The original training dataset contained:

| Class | Count   |
| ----- | ------- |
| 0     | 227,451 |
| 1     | 394     |

To address the imbalance problem, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied.

After SMOTE:

| Class | Count   |
| ----- | ------- |
| 0     | 227,451 |
| 1     | 227,451 |

This generated synthetic fraud samples and created a balanced training dataset.

---

## Models Used

### 1. Logistic Regression

A linear classification algorithm commonly used as a baseline model.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve classification performance.

---

## Evaluation Metrics

Accuracy was intentionally avoided because it can be misleading on highly imbalanced datasets.

The following metrics were used:

### Precision

Measures how many predicted fraud transactions were actually fraud.

### Recall

Measures how many actual fraud transactions were successfully detected.

### ROC-AUC

Measures the overall ability of the model to distinguish between fraudulent and legitimate transactions.

---

## Results

| Model               | Precision | Recall | ROC-AUC |
| ------------------- | --------- | ------ | ------- |
| Logistic Regression | 0.1304    | 0.8980 | 0.9745  |
| Random Forest       | 0.8351    | 0.8265 | 0.9644  |

---

## Best Model

### Random Forest Classifier

Although Logistic Regression achieved a slightly higher ROC-AUC score, Random Forest produced significantly higher precision while maintaining strong recall.

This makes Random Forest more suitable for fraud detection because it reduces false fraud alerts while still detecting most fraudulent transactions.

---

## Visualizations

The project generates:

* Class Distribution Plot
* Confusion Matrix
* ROC Curve
* Model Comparison Results

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)

---

## Project Workflow

1. Load Dataset
2. Analyze Class Distribution
3. Split Training and Testing Data
4. Apply SMOTE
5. Train Logistic Regression
6. Train Random Forest
7. Evaluate Models
8. Compare Results
9. Select Best Model

---

## Output Files

```text
creditcard.csv
project2.py
class_distribution.png
confusion_matrix.png
roc_curve.png
model_comparison.csv
README.md
```

---

## Conclusion

This project successfully developed a fraud detection pipeline capable of identifying fraudulent transactions in a highly imbalanced dataset. By applying SMOTE and evaluating models using Precision, Recall, and ROC-AUC, the project demonstrates practical machine learning techniques commonly used in financial fraud detection systems.
