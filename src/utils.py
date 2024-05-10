import numpy as np
import pandas as pd

def generate_uncorrelated_data(data):
    """
    Función que genera un clon de la misma forma que data pero donde cada columna se genera
    aleatoreamente siguiendo la misma distribución de cada columna asumiendo distribución normal.
    Se considera que la distribución de cada columna es independiente al resto

    Parámetros
    ----------
    data: array_like, numpy.ndarray o pandas.DataFrame.
        Datos de input
    
    Retorna
    -------
    data_resampled: numpy.ndarray o pandas.DataFrame.
        Datos re-sampleados aleatoreamente. Es el mismo datatype y forma que el input
    """
    data_np = np.array(data)
    # Inicializamos con una muestra aleatorea
    data_resampled = np.random.randn(*data.shape)
    # Multiplicamos por la desviación estándar y sumamos la media
    data_resampled = data_resampled * data_np.std(0) + data_np.mean(0)
    # Si el input era pandas, retornamos un pandas
    if isinstance(data, pd.DataFrame):
        data_resampled = pd.DataFrame(data_resampled, columns=data.columns)
    return data_resampled