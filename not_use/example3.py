import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from PIL import Image
from io import BytesIO

def main():

    prof = False
    auto_segments = 8
    n_clusters_max = 10
    # st.title("K-Means Segmentation with Stability Measures")
    st.title('Brain Food')
    image = Image.open('./images/01 version principal color.png')
    st.sidebar.image(image, width=150)
    st.sidebar.markdown('## Cargar Datos')
    uploaded_file = st.sidebar.file_uploader("Elegir archivo CSV", type="csv")

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)
        data['fullVisitorId'] = data['fullVisitorId'].astype(str)
        st.markdown('### Muestreo Datos')
        st.markdown('#### A continuación vemos una muestra aleatoria de los datos')
        st.write(data.sample(5))
        id_col = st.sidebar.selectbox('Elegir columna con identificador único', options=data.columns)
        # identificar columnas numericas para segmentación
        numeric_columns = data.select_dtypes(include=[int, float]).columns
        num_features = st.sidebar.multiselect('Seleccionar las variables numéricas relevantes', options=[c for c in data.columns], default=[c for c in numeric_columns if c != id_col])
        data = data.drop(id_col, axis=1) # dataframe sin id
        prof = st.checkbox('Marcar para analizar segmentos')

    else:
        st.markdown("""
            <br>
            <br>
            <h1 style="color:#26608e;"> Customer Analytics </h1>
            <h3 style="color:#106A73;"> Entendiendo al consumidor a través de los datos </h3>
        """, unsafe_allow_html=True)

    if (prof == True) & (uploaded_file is not None):

        # Create Segmentation object
        seg = Segmentation()
        data_norm = seg.normalize(data, num_features)
        # Number of clusters slider
        st.header("Cluster Settings")
        num_clusters = st.slider("Select the number of clusters", 2, n_clusters_max, 3)
        # Perform k-means segmentation and calculate stability measures
        seg.kmeans_segmentation(data_norm, num_clusters)
        print("f:Clusters labels: {seg.cluster_labels[0]}")

        # Display cluster plot
        pca_num_components = 2
        reduced_data = PCA(n_components=pca_num_components).fit_transform(data_norm)#(data_norm.iloc[:,1:12])
        results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

        width = st.sidebar.slider("plot width", 0.1, 25., 6.)
        height = st.sidebar.slider("plot height", 0.1, 25., 2.)

        fig, ax = plt.subplots(figsize=(width, height))
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=seg.cluster_labels, cmap='viridis')
        ax.legend()

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        # plt.figure(figsize=(4, 3))
        # fig, ax = plt.subplots()
        # ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=seg.cluster_labels, cmap='viridis')
        # st.pyplot(fig)
        #plt.scatter(results[:, 0], results[:, 1], c=seg.cluster_labels, cmap='viridis')
        #plt.title(f'K-Means Clustering (K={num_clusters})')
        #st.pyplot(plt)

        # Display stability measures
        st.header("Stability Measures")
        # st.write(f"Jaccard Index: {jaccard:.4f}")
        # st.write(f"Rand Index: {rand:.4f}")





if __name__ == "__main__":
    main()
