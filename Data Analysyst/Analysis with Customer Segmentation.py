import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Load Data
df = pd.read_csv('customer_data.csv')

# Data Cleaning and Preprocessing
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
df['days_since_last_purchase'] = (pd.Timestamp('today') - df['last_purchase_date']).dt.days
df = df.dropna()

# Feature Engineering
df['total_spend'] = df['purchase_count'] * df['average_purchase_value']
df['frequency'] = df.groupby('customer_id')['purchase_count'].transform('count')
df['recency'] = df['days_since_last_purchase']
df['monetary'] = df['total_spend']

# Feature Selection
features = ['frequency', 'recency', 'monetary', 'age', 'income', 'loyalty_score']
X = df[features]

# Feature Importance using Mutual Information
mi_scores = mutual_info_regression(X, df['total_spend'], random_state=0)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)
print(mi_scores)

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine Optimal Number of Clusters using Silhouette Score and Elbow Method
silhouette_scores = []
inertias = []

for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), silhouette_scores, 'bx-')
plt.title('Silhouette Analysis For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), inertias, 'bx-')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()

# Choose optimal k
optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(X_scaled)
df['cluster'] = kmeans.labels_

# Use PCA for Visualization
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
pca_df['cluster'] = df['cluster']

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['cluster'], cmap='viridis')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.colorbar(scatter)
plt.title('3D PCA Plot of Customer Segments')
plt.show()

# Analyze Segments
for i in range(optimal_clusters):
    cluster_data = df[df['cluster'] == i]
    print(f"Cluster {i} Summary:")
    print(cluster_data[features].describe())

    # Visualize feature distributions within each cluster
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.ravel()
    for j, feature in enumerate(features):
        sns.histplot(cluster_data[feature], ax=axes[j], kde=True)
        axes[j].set_title(f'{feature} Distribution in Cluster {i}')

# Statistical Tests for Cluster Differences
for feature in features:
    f_val, p_val = stats.f_oneway(*[df[df['cluster'] == i][feature] for i in range(optimal_clusters)])
    print(f"ANOVA for {feature}: F={f_val}, p={p_val}")

# Predictive Modeling for Cluster Behavior
rf = RandomForestRegressor(random_state=42)
rf.fit(X_scaled, df['total_spend'])
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances)

# Save the results
df.to_csv('segmented_customers_analysis.csv', index=False)
