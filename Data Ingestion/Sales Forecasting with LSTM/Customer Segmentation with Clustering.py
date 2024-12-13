#For customer segmentation, we'll use K-means clustering from Scikit-Learn, focusing on features like purchase history, demographics, and behavior

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load customer data
customer_data = pd.read_csv('customer_data.csv')

# Define features for segmentation
segmentation_features = ['total_purchases', 'avg_purchase_value', 'frequency_of_purchase', 'age', 'income']
X = customer_data[segmentation_features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine number of clusters (e.g., using the elbow method)
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
# Plot elbow curve to choose optimal k (not included here, but you'd plot inertias vs. k)

# Fit the model with chosen k (let's assume k=4 for this example)
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
for i in range(4):
    cluster = customer_data[customer_data['cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster[segmentation_features].describe())
    
# Visualize clusters if possible, e.g., for 2D data:
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=customer_data['cluster'], cmap='viridis')
plt.show()
