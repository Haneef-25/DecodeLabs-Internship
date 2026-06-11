import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

# Missing value handling
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)

plt.boxplot(df["Fare"])
plt.title("Fare Box Plot")
plt.ylabel("Fare")
plt.show() 


#IQR
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Fare"] = df["Fare"].clip(lower_bound, upper_bound)


outliers = df[
    (df["Fare"] < lower_bound) |
    (df["Fare"] > upper_bound)
]

print("Outliers after treatment:", len(outliers))


#Feature Engineering

#feature 1 --> Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
print(df[["SibSp","Parch","FamilySize"]].head())


#feature 2 --> is Alone or not
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
print(df[["FamilySize","IsAlone"]].head())  

#feature 3--> Age Categorization
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0,18,35,60,100],
    labels=["Child","Young","Adult","Senior"]
)
print(
    df[
        [
            "Age",
            "AgeGroup",
            "FamilySize",
            "IsAlone"
        ]
    ].head(10)
)


#one hot coding
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked", "AgeGroup"],
    drop_first=True
)

df.to_csv(
    "cleaned_titanic.csv",
    index=False
)

print("Dataset saved successfully!")