import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo
import os
from src.app_style.style_details import load_background_image
from src.gen_text import get_clusters_descriptive_text
from src.models.segmentation import Segmentation
from src.models.rfm import Rfm
from streamlit_extras.stateful_button import button
from src.streamlit_tools_aux import segment_data
from src.visualizations import draw_elbow_method

import os

# ===========================================================================================
# Design settings
# ===========================================================================================
with open( "app\style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

page_bg_img = load_background_image()
st.markdown(page_bg_img, unsafe_allow_html=True)

# add bf logo
add_logo("./images/bf_logo_small.png")
# ===========================================================================================
# End of design settings
# ===========================================================================================


def main():
    st.title('< Segmentaciones')
    # option button
    selected_opt = st.radio("Elija la fuente de datos a utilizar",["Datos de ejemplo", "Subir dataset propio"],
                            captions=["Se utilizará un dataset pre-definido", "Cargar dataset en formato .csv"],
                            index=1)
    if selected_opt == "Datos de ejemplo":
        data = pd.read_csv('./data/ga_customers.csv', encoding='unicode_escape')
        uploaded_file = True
    else:
        uploaded_file = st.file_uploader("Elegir archivo CSV", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file, encoding='unicode_escape')

    if uploaded_file is not None:
        st.markdown('### Muestreo Datos')
        st.markdown('#### A continuación vemos una muestra aleatoria de los datos')
        st.write(data.sample(5))
        st.markdown("""
        ##### \\\\\\\\ Paso 1
        Debes elegir la columna de la data que se utiliza como ID del cliente en la data. Esto es relevante ya que debe 
        excluirse al momento de ejecutar el modelo, pero será considerada al momento de identificar a los clientes y el 
        clúster al que pertenecen. En la data de ejemplo, el id corresponde a `fullVisitorId`.
        """)
        # Se eliminan columnas vacías en todo el procesamiento
        data = data[data.columns[(~data.isna()).sum()>0]]
        id_col = st.selectbox('Elegir columna con identificador único', options=data.columns)
        # identificar columnas numericas para segmentación
        st.markdown("""
            ##### \\\\\\\\ Paso 2
            La clusterización se realiza utilizando variables numéricas, excluyendo variables con formato string. A continuación,
            selecciona las **varaibles numéricas** que quieres considerar en el análisis. 
            """)
        numeric_columns = data.select_dtypes(include=[int, float]).columns
        num_features = st.multiselect('Seleccionar las variables numéricas relevantes',
                                              options=[c for c in data.columns],
                                              default=[c for c in numeric_columns if c != id_col])
        # data = data.drop(id_col, axis=1)  # Esto se movió a más adelante
        # RFM
        st.markdown("""
                ##### \\\\\\\\ Paso 3
                Adicionalmente a las variables relacionadas a los clientes, se pueden incluir datos relacionados a las 
                compras realizadas por ellos y extraer información de que tan Recientes, Frecuentes o que tanto es el
                aporte Monetario de estas (RFM). Para esto, se deben cargar los datos de las compras.
                """)
        transactions_file = st.file_uploader("Elegir archivo CSV de las compras (opcional)", type="csv")
        if transactions_file is not None:
            transactions = pd.read_csv(transactions_file, encoding='unicode_escape')
            st.markdown("""
                    ##### \\\\\\\\ Paso 3.2
                    Para utilizar los datos de las compras, se deben especificar las variables relacionadas a RFM.
                    """)
            cols_opcionales = ["---Ninguna---"] + transactions.columns.to_list()
            df2_id_cliente_col = st.selectbox('Elegir columna con el identificador del cliente', options=cols_opcionales)
            df2_date_col       = st.selectbox('Elegir columna con la fecha', options=cols_opcionales)
            # Si no se especifica id de transacción, se toma cada fila como una transacción distinta
            df2_id_trans_col   = st.selectbox('Elegir columna con el identificador de la transacción (opcional)', 
                                  options=cols_opcionales)
            df2_price_col  = st.selectbox('Elegir columna con el precio del producto (opcional)', 
                                  options=cols_opcionales)
            df2_amount_col  = st.selectbox('Elegir columna con la cantidad de productos comprados (opcional)', 
                                  options=cols_opcionales)
            df2_id_cliente_col = None if df2_id_cliente_col == "---Ninguna---" else df2_id_cliente_col
            df2_date_col       = None if df2_date_col       == "---Ninguna---" else df2_date_col
            df2_id_trans_col   = None if df2_id_trans_col   == "---Ninguna---" else df2_id_trans_col
            df2_price_col      = None if df2_price_col      == "---Ninguna---" else df2_price_col
            df2_amount_col     = None if df2_amount_col     == "---Ninguna---" else df2_amount_col
            #TODO: Agregar chequeos de si están bien las variables
            #TODO: Incluir selector de fecha mínima para filtrar
            #TODO: Seleccionar qué variables se calculan en el RFM
            if (df2_id_cliente_col is not None) and (df2_date_col is not None):
                rfm = Rfm(transactions, df2_id_cliente_col, df2_date_col, df2_id_trans_col, df2_amount_col,
                        df2_price_col)
                rfm.recency()
                rfm.frequency()
                rfm.monetary()
                data = rfm.export(data, id_col)
                num_features = num_features + rfm.columns
                if len(rfm.columns)>0:
                    st.markdown(f"Agregadas los atributos {', '.join(rfm.columns)} a la tabla. Si faltan atributos,\
                                seleccionar columnas adicionales para RFM")
                else:
                    st.markdown("ERROR: No se pudieron calcular los atributos de RFM")
            else:
                st.markdown("Seleccionar al menos la columna de id de cliente y fecha para calcular RFM")
        # Create Segmentation object
        st.markdown("""
                ##### \\\\\\\\ Paso 4
                Considerando que el algoritmo de segmentación realiza cálculos de distancia entre los puntos, un paso 
                importante previo a clusterizar es **normalizar las variables numéricas**. Esto último ayudará a que no
                adquieran más importancia variables que se encuentran en más magnitud
                """)
        data = data.drop(id_col, axis=1)  # dataframe sin id
        data[num_features] = data[num_features].fillna(data[num_features].mean()) # Se llenan los NaNs con el promedio

        norm_ = button("Normalizar data", key="button 1")

        if norm_ is True:
            seg = Segmentation()
            data_norm = seg.normalize(data, num_features)
            st.markdown("""
            ##### Vista de la data normalizada
            """)
            st.write(data_norm)
            st.markdown("""
                        ##### \\\\\\\\ Paso 5
                        Ahora que la data está lista, es posible realizar un análisis de los segmentos
                        """)
            # prof = st.button("Analizar segmentos", type="primary")
            print(data_norm.columns)
            prof = button("Analizar segmentos", key="button 2")
            if prof is True:
                model_choice = st.selectbox("Elige un algoritmo de clusterización: ",
                                            options=['kmeans', 'gmm', 'dbscan'])
                if model_choice in ("kmeans", "gmm"):
                    fig = draw_elbow_method(data=data_norm, seg_=seg)
                    st.image(fig)
                if model_choice == 'kmeans':
                    num_clusters = st.number_input("Cuántos clusters quieres definir:", min_value=2, max_value=6,
                                                   value=2)
                    seg.kmeans_segmentation(data_norm, num_clusters)
                elif model_choice == 'gmm':
                    num_clusters = st.number_input("Cuántos clusters quieres definir:", min_value=2, max_value=6,
                                                   value=2)
                    seg.gmm_segmentation(data_norm, num_clusters)
                elif model_choice == 'dbscan':
                    min_samples = st.number_input("Cuántos sampleos mínimos se necesitan por cluster:", min_value=5,
                                                  max_value=100, value=10)
                    eps = st.slider("Valor de Epsilon (eps) para DBSCAN: ", 0.01, 1.0, 0.1)
                    seg.dbscan_segmentation(data_norm, eps, min_samples)
                cluster_stats = segment_data(data_norm, data, seg, model_choice, num_features)
                st.write(cluster_stats)
                describe_clusters = button("Describir segmentos", key="button 2")
                if describe_clusters:
                    clusters_text = get_clusters_descriptive_text(cluster_stats)
                    st.write(clusters_text)


if __name__ == "__main__":
    main()
