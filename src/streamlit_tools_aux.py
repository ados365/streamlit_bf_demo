import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from bfstyle.load_bf_style import colors
import io
from src.models.segmentation import get_data_with_cluster, get_stats_table
from src.visualizations import cluster_3d_visual, cluster_3d_visual_discrete, get_stability_plots
import numpy as np


def select_id_column(data):
    return st.sidebar.selectbox('Elegir columna con identificador único', options=data.columns)


def select_numeric_features(data):
    numeric_columns = data.select_dtypes(include=[int, float]).columns
    return st.sidebar.multiselect('Seleccionar las variables numéricas relevantes',
                                  options=[c for c in data.columns],
                                  default=[c for c in numeric_columns])


def download_excel_multyple_sheets(df_dict, file_name):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    format1 = workbook.add_format({'num_format': '0.00'})
    for df_name in df_dict:
        if df_dict[df_name] is not None:  # and not df_dict[df_name].empty:
            df_dict[df_name].to_excel(writer, index=False, sheet_name=df_name)
            worksheet = writer.sheets[df_name]
            worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()

    st.download_button(
        label="Descargar archivo",
        data=processed_data,
        file_name=f"{file_name}.xlsx",
    )


def segment_data(data_norm, data_, _seg, model_choice_, num_features, pca_num_components=3):
    # Display cluster plot
    reduced_data = PCA(n_components=pca_num_components).fit_transform(data_norm)
    results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2', 'pca3'])
    # plot data
    # cluster_3d_visual(data=results, color1=colors[0], color2=colors[1], color3=colors[2], height_=600, seg_=_seg)
    cluster_3d_visual_discrete(data=results, height_=600, seg_=_seg)
    # Display stability measures
    st.write("""
    ### < Índices de Estabilidad
    El índice Jaccard y Rand son herramientas útiles para medir qué tan similares son dos conjuntos de datos de 
    segmentación. El índice Jaccard compara la cantidad de elementos comunes entre dos segmentaciones con la cantidad 
    total de elementos en ambas segmentaciones, proporcionando un valor que varía de 0 a 1, donde 1 indica una 
    similitud perfecta. Por otro lado, el índice Rand evalúa la cantidad de pares de elementos que son clasificados de 
    la misma manera o de manera diferente en ambas segmentaciones, dándonos una medida de acuerdo que va de 0 a 1, 
    donde 1 representa un acuerdo completo. Ambos índices son útiles para entender cuán cercanas están las segmentaciones 
    entre sí, siendo el índice Jaccard más adecuado para segmentaciones desbalanceadas y el índice Rand más robusto ante 
    segmentaciones con diferentes tamaños.
    """)
    rand, jaccard = _seg.stability_measures(data_norm, _seg.cluster_labels, model=model_choice_)
    fig, axs = get_stability_plots(rand, jaccard, color1=colors[0], color2=colors[6])
    st.pyplot(fig)
    # Display cluster statistics
    st.write("### < Estadísticas de los Clusters")
    cluster_stats = pd.DataFrame()
    cluster_stats['Cluster'] = np.unique(_seg.cluster_labels)
    cluster_stats['Num. Elementos'] = [np.sum(_seg.cluster_labels == i) for i in np.unique(_seg.cluster_labels)]
    # Show data frame with cluster
    st.write("##### \\\\\\ Tablas de datos con cluster")
    data_with_cluster = get_data_with_cluster(data_, _seg)
    st.write(data_with_cluster)
    st.write("##### \\\\\\ Tabla de estadísticas por cluster")
    stats_table = get_stats_table(data_with_cluster, col_name='cluster', stats_list=['mean', 'min', 'max', 'std'])
    st.write(stats_table)
    df_to_download_dict = {'resultados_cluster': data_with_cluster,
                           'estadisticas_cluster': stats_table}
    download_excel_multyple_sheets(df_dict=df_to_download_dict, file_name='resultados_segmentacion')
    st.write("##### \\\\\\ Descripción de clusters by LLM")

    for feature in num_features:
        cluster_stats[f'mean_{feature}'] = [np.mean(data_[_seg.cluster_labels == i][feature]) for i in
                                            np.unique(_seg.cluster_labels)]
    return cluster_stats
