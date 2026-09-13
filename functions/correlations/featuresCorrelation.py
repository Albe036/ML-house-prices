import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, chi2_contingency, association
import os

df = pd.read_csv("./data/raw/train.csv")

    

# ****************************************************************************************
# Theils U: también conocido como Coeficiente de Incertidumbre o Entropy Coefficient
# -------------------------------------------------
# H(X): Entropía de la variable X (su incertidumbre total).
# H(X|Y): Entropía de X dado que conocemos Y (incertidumbre restante).
# H(X) - H(X|Y): Es la Información Mutua entre X e Y.
# Interpretación directa: U(X|Y) responde a la pregunta: "Si conozco el valor de Y,
#  ¿en qué porcentaje se reduce mi incertidumbre sobre X?"
# ****************************************************************************************
class Features_correlations_theils_u(FeaturesHandling):
    pass

class FeaturesCorrelations(
    FeaturesHandling, Features_correlations_pearson, Features_correlations_spearman
):
    def difference_between_pearson_and_spearman(self, featureAnalyze):
        numeric_cols = super()._filter_features(type="numeric")
        cols = numeric_cols.columns.tolist()
        if featureAnalyze not in cols:
            raise ValueError(
                "featureAnalyze must be numeric and present in the DataFrame"
            )
        res = []
        for col in cols:
            pearson_stat, p_value_pearson = pearsonr(
                numeric_cols[featureAnalyze], numeric_cols[col]
            )
            spearman_stat, p_value_spearman = spearmanr(
                numeric_cols[featureAnalyze], numeric_cols[col]
            )
            difference = abs(pearson_stat - spearman_stat)
            # difference > 0.2
            # ALERTA: Diferencia significativa entre Pearson y Spearman.
            # Esto indica que la relación NO es estrictamente lineal.
            # Spearman es mas fiable para esta prueba.
            # difference <= 0.2
            # Pearson y Spearman son consistentes.
            # La relación es aproximadamente lineal.
            res.append(
                {
                    "featureAnalyze": featureAnalyze,
                    "featureCorrelation": col,
                    "pearson_stat": pearson_stat,
                    "p_value_pearson": p_value_pearson,
                    "spearman_stat": spearman_stat,
                    "p_value_spearman": p_value_spearman,
                    "difference": difference,
                }
            )
        return super()._config_output(res)
