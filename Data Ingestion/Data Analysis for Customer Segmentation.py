import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load Data
df = pd.read_csv('customer_data.csv')

# Data Cleaning and Preprocessing
df['last_purchase_days'] = (pd.to_datetime('today') - pd.to_datetime(df['last_purchase_date'])).dt.days
df = df.dropna()

# Feature Engineering
df['frequency'] = df.groupby('customer_id')['transaction_id'].transform('count')
df['monetary'] = df.groupby('customer_id')['amount_spent'].transform('sum')
df['recency'] = df['last_purchase_days']

# Select Features for Clustering
features = ['frequency', 'monetary', 'recency']
X = df[features]

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine Optimal Number of Clusters
silhouette_scores = []
for n_clusters in range(2, 11):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis For Optimal k')
plt.show()

# Apply KMeans with the best number of clusters
optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(X_scaled)
df['cluster'] = kmeans.labels_

# Use PCA for Visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
principal_df['cluster'] = kmeans.labels_

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=principal_df, palette='viridis')
plt.title('Customer Segments in PCA Space')
plt.show()

# Analyze Clusters
for i in range(optimal_clusters):
    cluster_data = df[df['cluster'] == i]
    print(f"Cluster {i} Summary:")
    print(cluster_data[features].describe())
    
    # Visualize distribution within each cluster
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for j, feature in enumerate(features):
        sns.histplot(cluster_data[feature], ax=ax[j], kde=True)
        ax[j].set_title(f'{feature} Distribution in Cluster {i}')

# Statistical Tests for Cluster Differences
for feature in features:
    f_val, p_val = stats.f_oneway(*[df[df['cluster'] == i][feature] for i in range(optimal_clusters)])
    print(f"ANOVA for {feature}: F={f_val}, p={p_val}")

# Save the results
df.to_csv('segmented_customers.csv', index=False)
