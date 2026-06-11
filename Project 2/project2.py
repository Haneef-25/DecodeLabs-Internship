import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from imblearn.over_sampling import SMOTE


# LOAD DATASET
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Class"].value_counts())


# CLASS DISTRIBUTION VISUALIZATION
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution")
plt.savefig("class_distribution.png")
plt.show()


# FEATURES AND TARGET
X = df.drop("Class", axis=1)
y = df["Class"]


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Size:", X_train.shape)
print("Testing Size:", X_test.shape)

print("\nBefore SMOTE:")
print(y_train.value_counts())


# APPLY SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# LOGISTIC REGRESSION
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_smote,
    y_train_smote
)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_prob_lr)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")

print("Precision:", lr_precision)
print("Recall:", lr_recall)
print("ROC-AUC:", lr_auc)



# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("ROC-AUC:", rf_auc)



# MODEL COMPARISON
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Precision": [
        lr_precision,
        rf_precision
    ],
    "Recall": [
        lr_recall,
        rf_recall
    ],
    "ROC-AUC": [
        lr_auc,
        rf_auc
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(comparison)

comparison.to_csv(
    "model_comparison.csv",
    index=False
)


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()



# ROC CURVE
fpr, tpr, _ = roc_curve(
    y_test,
    y_prob_rf
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {rf_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()


# BEST MODEL
if rf_auc > lr_auc:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Logistic Regression")