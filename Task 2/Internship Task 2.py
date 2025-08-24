import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")
print(df)

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

kmeans = KMeans(n_clusters= 5, random_state= 42)
df["Clusters"] = kmeans.fit_predict(X_scaler)

inertia = []
silhouette = []
K = range (2, 11)

for k in K:
    kmeans = KMeans (n_clusters= k, random_state= 42)
    labels = kmeans.fit_predict(X_scaler)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaler, labels))

plt.plot(K, inertia, "bo-")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

plt.plot(K, silhouette, "ro-")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c= df["Clusters"], cmap= "viridis", s= 50)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")
plt.colorbar(label = "Clusters")
plt.show()

dbscan = DBSCAN(eps= 0.5, min_samples= 5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaler)

plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c= df["DBSCAN_Cluster"], cmap= "plasma", s= 50)
plt.title("DBSCAN Clusters")
plt.colorbar(label = "DBSCAN_Cluster")
plt.show()

Cluster_avg = df.groupby("Clusters")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
print("\nAverage Spending per KMeans cluster:\n", Cluster_avg)

dbscan_avg = df[df["DBSCAN_Cluster"] != -1].groupby("DBSCAN_Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
print("\nAverage Spending per DBSCAN cluster:\n", dbscan_avg)