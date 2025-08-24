# 🛍️ Customer Segmentation (Clustering)

This project applies **unsupervised learning** techniques to segment mall customers based on their annual income and spending score.

---

## 🔧 Libraries Used
- `pandas`  
- `scikit-learn`  
- `matplotlib`  

---

## 📂 Dataset
**Mall_Customers.csv**  
- 200 rows  
- 5 features: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1–100)`  

---

## 🔹 Steps Performed

### 1. Preprocessing
- Selected **Annual Income (k$)** and **Spending Score (1–100)**  
- Standardized features using **StandardScaler**  

### 2. KMeans Clustering
- Applied KMeans clustering (`n_clusters=5`)  
- Determined optimal K using **Elbow Method** and **Silhouette Score**  
- Visualized results in scatter plots  

### 3. DBSCAN Clustering (Bonus)
- Applied **DBSCAN** to detect dense regions & noise  
- Compared results with KMeans  

### 4. Cluster Insights
- Calculated **average income & spending score** per cluster  

**Example (KMeans):**
- Cluster 1 → High Income, High Spending  
- Cluster 2 → High Income, Low Spending  
- Cluster 3 → Low Income, High Spending  
- Cluster 4 → Low Income, Low Spending  
- Cluster 5 → Moderate Income, Moderate Spending  

---

## 📊 Model Evaluation
Since clustering is **unsupervised learning**, accuracy cannot be measured directly. Instead, the following metrics were used:

- 🔹 **Elbow Method (Inertia):** Identified optimal cluster count  
- 🔹 **Silhouette Score:** Ranged between `0.40–0.55`, suggesting reasonably good clustering  
- 🔹 **Visual Inspection:** Scatter plots showed clear separation  
- 🔹 **Cluster Averages:** Helped interpret customer groups  

---

## ✅ Results
- KMeans → Segmented customers into **5 groups**  
- DBSCAN → Found **2 main groups + noise points**  
- Cluster analysis revealed distinct customer types (e.g., **high spenders vs. low spenders**)  

---

✨ This project demonstrates how **unsupervised learning** can provide valuable business insights such as **targeted marketing** and **customer profiling**.
