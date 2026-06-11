# Project 1: Advanced EDA & Feature Engineering

## Overview

This project focuses on transforming raw data into a clean and machine-learning-ready dataset using Exploratory Data Analysis (EDA), data preprocessing, outlier treatment, and feature engineering techniques.

The Titanic dataset was used to demonstrate the complete data cleaning workflow, including handling missing values, detecting and treating outliers, creating new features, and encoding categorical variables.

---

## Objectives

* Perform Exploratory Data Analysis (EDA) on a real-world dataset.
* Handle missing values using statistical techniques.
* Detect and treat outliers using the Interquartile Range (IQR) method.
* Create meaningful features from existing data.
* Convert categorical data into numerical format using One-Hot Encoding.
* Generate a cleaned dataset suitable for machine learning applications.

---

## Dataset

**Dataset:** Titanic Passenger Dataset

The dataset contains passenger information such as:

* Passenger Class (Pclass)
* Name
* Gender (Sex)
* Age
* Number of Siblings/Spouses (SibSp)
* Number of Parents/Children (Parch)
* Fare
* Cabin
* Embarkation Port (Embarked)
* Survival Status

---

## Data Preprocessing

### Missing Value Handling

The dataset contained missing values in the following columns:

| Column   | Missing Values | Action Taken         |
| -------- | -------------- | -------------------- |
| Age      | 177            | Filled using Median  |
| Embarked | 2              | Filled using Mode    |
| Cabin    | 687            | Dropped from dataset |

**Reasoning:**

* Median was chosen for Age because it is less sensitive to outliers.
* Mode was used for Embarked as it is a categorical feature.
* Cabin was removed due to a very high percentage of missing values.

---

## Outlier Detection and Treatment

Outliers were detected in the **Fare** column using the **Interquartile Range (IQR)** method.

### Formula

IQR = Q3 − Q1

Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

### Treatment

Instead of removing rows, outliers were treated using **Winsorization** through the `clip()` function, which caps extreme values within acceptable limits while preserving all records.

---

## Feature Engineering

Three new features were created:

### 1. FamilySize

Represents the total number of family members traveling together.

```python
FamilySize = SibSp + Parch + 1
```

### 2. IsAlone

Indicates whether a passenger was traveling alone.

```python
IsAlone = 1 if FamilySize == 1 else 0
```

### 3. AgeGroup

Categorizes passengers into age-based groups:

* Child (0–18)
* Young (18–35)
* Adult (35–60)
* Senior (60+)

These engineered features provide additional information that may improve predictive performance.

---

## Categorical Encoding

Categorical columns were transformed into numerical format using **One-Hot Encoding**.

Encoded Columns:

* Sex
* Embarked
* AgeGroup

This step ensures compatibility with machine learning algorithms that require numerical inputs.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn

---

## Project Workflow

1. Load Dataset
2. Perform Exploratory Data Analysis (EDA)
3. Handle Missing Values
4. Detect Outliers using IQR
5. Treat Outliers using Winsorization
6. Create New Features
7. Apply One-Hot Encoding
8. Save Cleaned Dataset

---

## Output

The final processed dataset is saved as:

```text
cleaned_titanic.csv
```

This dataset is free from missing values, contains treated outliers, includes engineered features, and is ready for machine learning applications.

---

## Conclusion

This project demonstrates the importance of data preprocessing and feature engineering in the data science workflow. By handling missing values, treating outliers, and creating meaningful features, the quality of the dataset was significantly improved. The resulting dataset is structured, consistent, and suitable for further predictive modeling tasks.
