import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, jaccard_score
import matplotlib.pyplot as plt

# Create synthetic data
n_samples = 300
n_features = 2
n_clusters_max = 10
random_state = 42
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=random_state)

st.title("K-Means Clustering with Stability Metrics")

# Slider to select the number of clusters
n_clusters = st.slider("Select the number of clusters", 2, n_clusters_max, 3)

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans.fit(X)
labels = kmeans.labels_

# Scatter plot of the clustered data
st.subheader("Clustered Data")
st.subheader("Clustered Data")
plt.figure(figsize=(10, 6))
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
#plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
st.pyplot(fig)

# Calculate Rand and Jaccard metrics for stability evaluation
rand_scores = []
jaccard_scores = []
for _ in range(10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=np.random.randint(0, 100))
    kmeans.fit(X)
    new_labels = kmeans.labels_
    
    rand_score = adjusted_rand_score(labels, new_labels)
    jaccard = jaccard_score(labels, new_labels, average="macro")
    
    rand_scores.append(rand_score)
    jaccard_scores.append(jaccard)

st.subheader("Stability Metrics")
st.write(f"Average Rand Score: {np.mean(rand_scores):.3f}")
st.write(f"Average Jaccard Score: {np.mean(jaccard_scores):.3f}")
