import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from IPython.display import display as dsp
import os

df = pd.read_csv("./data/raw/train.csv")


# -------------------------------------------------
# Pearson Correlation Test
# El coeficiente de Pearson, mide la relación lineal entre dos variables cuantitativas.
#   - Valores: Va de -1 (correlación negativa perfecta) a +1 (correlación positiva perfecta).
#   - Cero (0): Significa ausencia de relación lineal, pero NO significa que son independientes
#   (pueden tener una relación cuadrática perfecta y Pearson dará 0)
# Interpretación:
# r > 0.9 → Muy alta (redundancia fuerte)
# r > 0.7 → Alta (redundancia)
# r > 0.5 → Moderada
# r < 0.3 → Baja
# -------------------------------------------------
class Features_correlations:

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

    def __filter_features(self, type="numeric", custom_features=[]):
        if len(custom_features) > 0:
            custom_features = [
                f for f in custom_features if f in self.dataFrame.columns
            ]
            if type == "numeric":
                return (
                    self.dataFrame[custom_features]
                    .select_dtypes(include=[np.number])
                    .dropna()
                )
            else:
                return (
                    self.dataFrame[custom_features]
                    .select_dtypes(exclude=[np.number])
                    .dropna()
                )
        else:
            if type == "numeric":
                return self.dataFrame.select_dtypes(include=[np.number]).dropna()
            else:
                return self.dataFrame.select_dtypes(exclude=[np.number]).dropna()

    def calculate_pearson_matrix(self, custom_features=[], interpret=False):
        numeric_features = self.__filter_features(
            type="numeric", custom_features=custom_features
        )
        matrix = numeric_features.corr(method="pearson").round(2)
        if interpret:
            return matrix.apply(lambda x: x.apply(self.__interpret_pearson_value))
        return matrix

    def __interpret_pearson_value(self, r):
        if r == 1:
            return "--"
        elif r > 0.9:
            return "Very High"
        elif r > 0.7:
            return "High"
        elif r > 0.5:
            return "Moderate"
        elif r < 0.3:
            return "Low"
        else:
            return "Moderate"


fc = Features_correlations(df)
res = fc.calculate_pearson_matrix([], interpret=True)
print(res)
