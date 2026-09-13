import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from functions.correlations.FeaturesHandling import FeaturesHandling

# ****************************************************************************************
# Spearman Correlation test
# Es una medida que indica qué tan relacionadas están dos variables según su orden o posición, no según sus valores exactos.
# Para calcularlo, se reemplazan los valores por sus rangos (1.º, 2.º, 3.º, etc.) y se mide si esos rangos suben o bajan juntos.
# Da valores entre -1 y +1.
# +1: cuando una variable sube, la otra siempre sube (en el mismo orden).
# -1: cuando una variable sube, la otra siempre baja.
# 0: no hay relación de orden entre ellas.
# -------------------------------------------------
# Su ventaja es que no le afectan los valores extremos y no exige que los datos sean normales.
# Su límite es que no detecta relaciones que suben y luego bajan (como una U o una campana).
# -------------------------------------------------
# Interpretation: (abs(r))
# r > 0.8 -> Very High
# 0.6 < r <= 0.8 -> High
# 0.4 < r <= 0.6 -> Moderate
# 0.1 < r <= 0.4 -> Low
# r <= 0.1 -> Insignificant
# ****************************************************************************************
class Features_correlations_spearman(FeaturesHandling):
    def all_features(self, featureAnalyze="", onlyTrue=True, desc=False):
        # H_0 = No hay correlación entre featureAnalyze y las demás características numéricas
        # H_1 = Existe correlación entre featureAnalyze y al menos una de las demás características numéricas
        numeric_features = super()._filter_features(type_features="numeric")
        cols = numeric_features.columns.tolist()
        if featureAnalyze not in cols:
            raise ValueError("feature must be numeric and present in the DataFrame")
        cols.remove(featureAnalyze)
        res = []
        for col in cols:
            spearman_stat, p_value = spearmanr(
                numeric_features[featureAnalyze], numeric_features[col]
            )
            res.append(
                {
                    "feature": col,
                    "spearman_stat": (
                        spearman_stat
                        if desc
                        else self._interpret_spearman_value(spearman_stat)
                    ),
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                    "res": f"{p_value} - {self.alpha}"
                }
            )
        return super()._config_output(res)

    def one_to_one(
        self, featureAnalyze, featureCorrelation, with_plot=True
    ):
        # H_0: No hay correlación entre featureAnalyze y featureCorrelation
        # H_1: Existe correlación entre featureAnalyze y featureCorrelation
        numeric_features = super()._filter_features(type="numeric")
        numeric_cols = numeric_features.columns.tolist()
        if featureAnalyze not in numeric_cols or featureCorrelation not in numeric_cols:
            raise ValueError(
                "Both features must be numeric and present in the DataFrame"
            )
        spearman_stat, p_value = spearmanr(
            numeric_features[featureAnalyze], numeric_features[featureCorrelation]
        )
        # plot correlation
        if with_plot:
            sns.lmplot(
                x=featureAnalyze, y=featureCorrelation, data=numeric_features, ci=None
            )
            plt.title(
                f"Spearman Correlation between {featureAnalyze} and {featureCorrelation}"
            )
            plt.xlabel(featureAnalyze)
            plt.ylabel(featureCorrelation)
            plt.show()
        return super()._config_output(
            [
                {
                    "featureAnalyze": featureAnalyze,
                    "featureCorrelation": featureCorrelation,
                    "spearman_stat": spearman_stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                }
            ]
        )

    def create_matrix(self, custom_features=[], interpret=False):
        numeric_features = super()._filter_features(
            type_features="numeric", custom_features=custom_features
        )
        matrix = numeric_features.corr(method="spearman").round(2)
        if interpret:
            return matrix.apply(lambda x: x.apply(self._interpret_spearman_value))
        return matrix
        # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    def _interpret_spearman_value(self, r):
        r_abs = abs(r)
        if r_abs == 1:
            return "--"
        elif r_abs >= 0.8:
            return "Very High"
        elif r_abs >= 0.6:
            return "High"
        elif r_abs >= 0.4:
            return "Moderate"
        elif r_abs >= 0.2:
            return "Low"
        else:
            return "Insignificant"
