import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, jaccard_score

# Define the Segmentation class
class Segmentation:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def segment(self):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        labels = kmeans.fit_predict(self.data)
        return labels

    @staticmethod
    def calculate_stability(labels1, labels2):
        jaccard = jaccard_score(labels1, labels2)
        rand = adjusted_rand_score(labels1, labels2)
        return jaccard, rand

# Streamlit app
def main():
    st.title("K-Means Segmentation with Stability Measures")

    # Generate sample data
    data, true_labels = make_blobs(n_samples=300, centers=3, random_state=0, cluster_std=1.0)

    # Number of clusters slider
    num_clusters = st.slider("Select the number of clusters:", 2, 10, 3)

    # Create a Segmentation instance
    seg = Segmentation(data, num_clusters)

    # Perform segmentation
    segmented_labels = seg.segment()

    # Calculate stability measures
    jaccard, rand = seg.calculate_stability(true_labels, segmented_labels)

    # Display the clusters
    plt.scatter(data[:, 0], data[:, 1], c=segmented_labels, cmap='viridis')
    plt.title(f'K-Means Segmentation (Jaccard={jaccard:.2f}, Rand={rand:.2f})')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(plt)

if __name__ == "__main__":
    main()
