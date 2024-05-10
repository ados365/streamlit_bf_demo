import numpy as np
import pandas as pd

class Rfm:
    def __init__(self, transactions_df, col_id_user, col_date, col_id_trans=None,
                 col_amount=None, col_price=None, min_date=None):
        # Dataframe de transacciones
        self.df=transactions_df.copy()
        # Columna con el id del usuario que realiza transaccion
        self.col_id_user=col_id_user
        # Columna con la fecha
        self.col_date=col_date
        # Columna el id de la transacción
        self.col_id_trans=col_id_trans
        # Columna con la cantidad de artículos comprados
        self.col_amount=col_amount
        # Columna con el precio del artículo comprado
        self.col_price=col_price
        # Transformarmos la fecha a datetime en caso de que no esté en ese formato
        self.df[col_date] = pd.to_datetime(self.df[col_date])
        # Fecha desde la que se considerarán las transacciones. Si no se da, se consideran todas
        self.min_date =self.df[col_date].min() if (min_date is None) else min_date
        # Última fecha del dataframe para calcular recencia
        self.max_date=self.df[col_date].max()
        self.df = self.df[self.df[col_date] >= self.min_date]
        # Filtramos el dataset para incluir sólo las columnas requeridas
        self.columnas = [col for col in (
            col_id_user, col_date, col_id_trans, col_amount, col_price
            ) if col is not None]
        self.df = self.df[self.columnas]
        # Se eliminan nulos de las columnas a usar
        self.df = self.df.dropna()
        # Se inicializa el dataframe a exportar
        self.df_grouped = pd.DataFrame(index = self.df[col_id_user].unique())
    # TODO: Agregar preprocesamiento
    def recency(self):
        self.df["DAYS_PASSED"] = (self.max_date - self.df[self.col_date]).dt.days
        self.df_grouped["RECENCY"] = self.df.groupby(self.col_id_user)["DAYS_PASSED"].min()
    def frequency(self):
        if self.col_id_trans is None:
            self.df_grouped["FREQUENCY"] = self.df.groupby(self.col_id_user)[self.col_date].count()
        else:
            self.df_grouped["FREQUENCY"] = self.df.groupby(self.col_id_user)[self.col_id_trans].nunique()
    def monetary(self):
        if (self.col_amount is not None) and (self.col_price is not None):
            self.df["TOTAL_AMOUNT"] = self.df[self.col_amount] * self.df[self.col_price]
            self.df_grouped["MONETARY"] = self.df.groupby(self.col_id_user)["TOTAL_AMOUNT"].sum()
    def export(self, df_users, id_col):
        joined =  df_users.join(self.df_grouped, on= id_col)
        if "RECENCY" in self.df_grouped.columns:
            joined["RECENCY"] = joined["RECENCY"].fillna(joined["RECENCY"].max())
        if "FREQUENCY" in self.df_grouped.columns:
            joined["FREQUENCY"] = joined["FREQUENCY"].fillna(0)
        if "MONETARY" in self.df_grouped.columns:
            joined["MONETARY"] = joined["MONETARY"].fillna(0)
        self.columns = self.df_grouped.columns.to_list()
        return joined
    