import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from bfstyle.load_bf_style import colors
from src.utils import generate_uncorrelated_data

colors_reordered = [colors[0], colors[7], *colors[2:6], colors[1], colors[6]]

def change_plot_color(color):
    plt.rcParams["axes.edgecolor"] = color
    plt.rcParams["axes.labelcolor"] = color
    plt.rcParams["text.color"] = color
    plt.rcParams["xtick.color"] = color
    plt.rcParams["ytick.color"] = color


def transparent_plot(fig):
    fig.savefig("images/cache.png", transparent=True, bbox_inches="tight")
    fig = Image.open("images/cache.png")
    return fig


def create_colormap(colors, docks, name="", default = False):
    """
    Función que crea un mapa de calor y lo registra
    Parámetros
    ----------
    colors: lista o np.ndarray de colores en formato string (matplotlib) o rgba (n_colores, 4)
        Lista que contiene los colores a utilizar para el colormap. Cada color es un tuple de 4 ints
        en formato RGBA en el rango 0-255, o un string en formato hex o con colores con nombre.
    docks: lista de floats de forma (n_colores, )
        Lista que contiene las posiciones en el rango 0-1 de cada uno de los colores de colors.
    name: str, default = ""
        Nombre que tendrá el colormap
    default: bool, default = False
        Si default = True, el colormap será el por defecto de matplotlib
    """
    cmap = np.zeros((256, 4))
    points = np.linspace(0, 1, 256)
    if isinstance(colors[0], str):
        colors = np.array([mpl.colors.to_rgba(color) for color in colors])
    colors = np.array(colors)
    for i in range(1, len(colors)):
        condition = np.all((points >= docks[i-1], points<= docks[i]), axis=0)
        cmap[condition] =  colors[i-1] + (points[condition]-docks[i-1]).reshape(-1, 1) * (
            colors[i]- colors[i-1]) / (docks[i]-docks[i-1])
    cmap = LinearSegmentedColormap.from_list(name = name, colors=cmap)
    mpl.colormaps.unregister(name)
    mpl.colormaps.register(cmap=cmap)
    if default:
        mpl.rc('image', cmap=name)
    return cmap


def cluster_3d_visual(data,color1,color2,color3,height_,seg_):
    fig = px.scatter_3d(data, x='pca1', y='pca2', z='pca3', color=seg_.cluster_labels,
                        title='Visualización clúster en 3D',
                        color_continuous_scale=[[0, color1], [0.5, color2], [1.0, color3]])
    fig.update_layout(height=height_)
    return st.plotly_chart(fig, theme=None, use_container_width=True, height=height_)


def cluster_3d_visual_discrete(data, height_, seg_):
    clusters = ["Cluster" + str(cluster+1) for cluster in seg_.cluster_labels]
    fig = px.scatter_3d(data, x='pca1', y='pca2', z='pca3', color=clusters,
                        title='Visualización clúster en 3D',
                        color_discrete_sequence=colors_reordered)
    fig.update_layout(height=height_)
    return st.plotly_chart(fig, theme=None, use_container_width=True, height=height_)


def draw_elbow_method(data, seg_):
    # Generamos datos aleatorios sin correlación para comparar los errores en el método del codo
    data_uncorr = generate_uncorrelated_data(data)
    # Calculamos el error con 2 a 10 clusters en los datos reales y random para el método del codo
    nums_clusters, sums_squares = seg_.elbow_method(data, 10)
    nums_clusters_rand, sums_squares_rand = seg_.elbow_method(data_uncorr, 10)
    # Gráfico blanco bonito
    change_plot_color("white")
    plt.rcParams["legend.labelcolor"] = "black"
    fig = plt.figure(figsize=(8, 3))
    plt.plot(nums_clusters, sums_squares, color=colors_reordered[0], label="Data real")
    plt.plot(nums_clusters_rand, sums_squares_rand, color=colors_reordered[1], label="Data random")
    plt.legend()
    plt.xlabel("N° Clusters")
    plt.ylabel("Error intra cluster")
    fig = transparent_plot(fig)
    return fig


def get_stability_plots(rand, jaccard,color1,color2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Gráfico de donut para el Jaccard Index
    sizes_jaccard = [jaccard, 1 - jaccard]
    axs[0].pie(
        sizes_jaccard,
        labels=['', ''],
        colors=[color1, 'lightgray'],
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )
    axs[0].set_title('Jaccard Index', fontsize=20, pad=20, loc='left')
    axs[0].axis('equal')

    axs[0].text(
        0, 0,
        f"{jaccard * 100:.1f}%",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        color='black'
    )

    axs[0].add_artist(plt.Circle((0, 0), 0.70, fc='white', ec='lightgray'))

    # Gráfico de donut para el Rand Index
    sizes_rand = [rand, 1 - rand]  # Se resta rand de 1 para tener la porción vacía
    axs[1].pie(
        sizes_rand,
        labels=['', ''],
        colors=[color2, 'lightgray'],
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )
    axs[1].set_title('Rand Index', fontsize=20, pad=20, loc='left')
    axs[1].axis('equal')

    # Agregar texto en el centro para mostrar el valor del Rand Index
    axs[1].text(
        0, 0,
        f"{rand * 100:.1f}%",
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        color='black'
    )

    axs[1].add_artist(plt.Circle((0, 0), 0.70, fc='white', ec='lightgray'))

    plt.legend().set_visible(False)

    return fig, axs


def get_pca_plot(reduced_data, clusters):

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', label=clusters)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(*sc.legend_elements(), title='Clusters')
    return fig, ax