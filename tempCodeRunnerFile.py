import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Pilih fitur yang akan digunakan untuk clustering
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# METODE ELBOW
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Visualisasi Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan K Optimal')
plt.grid(True)
plt.show()

# METODE SILHOUETTE SCORE
silhouette_scores = []
k_range_silhouette = range(2, 10)  # Mulai dari 2 karena silhouette tidak valid untuk 1 cluster
for k in k_range_silhouette:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Visualisasi Silhouette Score
plt.figure(figsize=(8, 5))
plt.plot(k_range_silhouette, silhouette_scores, marker='o', color='green')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score untuk Menentukan K Optimal')
plt.grid(True)
plt.show()

# CLUSTERING AKHIR (misalnya jumlah optimal adalah 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# VISUALISASI CLUSTERING

# Grafik 1: Clustering Age vs Income
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df['Age'],
    y=df['Annual Income (k$)'],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)
plt.title('Hasil Clustering Pelanggan Mall')
plt.xlabel('Usia')
plt.ylabel('Pendapatan Tahunan')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Grafik 2 Clustering Age & Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=df['Age'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='viridis',
    s=100
)
plt.title('Clustering Usia & Skor Pengeluaran')
plt.xlabel('Usia')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()