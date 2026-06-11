# Customer Segmentation using PCA and KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print(df.head())

# Data Preprocessing

# Convert Gender to numeric
df["Gender"] = df["Gender"].map({"Male":0,"Female":1})

X = df.drop("CustomerID", axis=1)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)


# Elbow Method
wcss = []

for k in range(1,11):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_pca)

    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# Silhouette Score
scores = []

for k in range(2,11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_pca)

    score = silhouette_score(X_pca, labels)

    scores.append(score)

plt.figure(figsize=(8,5))
plt.plot(range(2,11), scores, marker='o')
plt.title("Silhouette Scores")
plt.xlabel("Clusters")
plt.ylabel("Score")
plt.show()

best_k = scores.index(max(scores)) + 2

print("Best K =", best_k)


# Final KMeans
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_pca)

df["Cluster"] = clusters


# Visualization
plt.figure(figsize=(10,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap="viridis"
)

plt.title("Customer Segmentation")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.show()


# Cluster Analysis
persona = df.groupby("Cluster").mean()

print(persona)