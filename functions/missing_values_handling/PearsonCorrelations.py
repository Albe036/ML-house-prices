import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from functions.missing_values_handling.FeaturesHandling import FeaturesHandling


# ****************************************************************************************
# Pearson Correlation Test
# El coeficiente de Pearson, mide la relación lineal entre dos variables cuantitativas.
#   - Valores: Va de -1 (correlación negativa perfecta) a +1 (correlación positiva perfecta).
#   - Cero (0): Significa ausencia de relación lineal, pero NO significa que son independientes
#   (pueden tener una relación cuadrática perfecta y Pearson dará 0)
# -------------------------------------------------
# Pearson es extremadamente sensible a valores extremos (outliers) y a relaciones no lineales
# Si tus variables tienen forma de U o de campana, Pearson dará cercano a 0.
# Solución: Calcula también la Correlación de Spearman (que mide relaciones monótonas, no solo lineales).
# -------------------------------------------------
# Interpretación:
# r > 0.9 → very high (redundancia fuerte)
# 0.7 < r ≤ 0.9 → high (redundancia)
# 0.5 < r ≤ 0.7 → moderate
# 0.3 < r ≤ 0.5 → low
# r ≤ 0.3 → very low or negligible
# ****************************************************************************************
class PearsonCorrelations(FeaturesHandling):
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
            pearson_stat, p_value = pearsonr(
                numeric_features[featureAnalyze], numeric_features[col]
            )
            res.append(
                {
                    "feature": col,
                    "pearson_stat": (
                        pearson_stat
                        if desc
                        else self._interpret_pearson_value(pearson_stat)
                    ),
                    "p_value": p_value,
                    "evidence_MAR": (p_value < self.alpha),
                    "res": f"{p_value} - {self.alpha}",
                }
            )
        return super()._config_output(res, onlyTrue=onlyTrue)

    def one_to_one(
        self, featureAnalyze, featureCorrelation, with_plot=True, desc=False
    ):
        # H_0: No hay correlación entre featureAnalyze y featureCorrelation
        # H_1: Existe correlación entre featureAnalyze y featureCorrelation
        numeric_features = super()._filter_features(type_features="numeric")
        numeric_cols = numeric_features.columns.tolist()
        if featureAnalyze not in numeric_cols or featureCorrelation not in numeric_cols:
            raise ValueError(
                "Both features must be numeric and present in the DataFrame"
            )
        pearson_stat, p_value = pearsonr(
            numeric_features[featureAnalyze], numeric_features[featureCorrelation]
        )
        # plot correlation
        if with_plot:
            sns.lmplot(
                x=featureAnalyze, y=featureCorrelation, data=numeric_features, ci=None
            )
            plt.title(
                f"Pearson Correlation between {featureAnalyze} and {featureCorrelation}"
            )
            plt.xlabel(featureAnalyze)
            plt.ylabel(featureCorrelation)
            plt.show()
        return super()._config_output(
            [
                {
                    "featureAnalyze": featureAnalyze,
                    "featureCorrelation": featureCorrelation,
                    "pearson_stat": self._interpret_pearson_value(pearson_stat),
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                }
            ]
        )

    def create_matrix(self, custom_features=[], interpret=False):
        numeric_features = super()._filter_features(
            type_features="numeric", custom_features=custom_features
        )
        matrix = numeric_features.corr(method="pearson").round(2)
        if interpret:
            return matrix.apply(lambda x: x.apply(self._interpret_pearson_value))
        return matrix
        # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    def _interpret_pearson_value(self, r):
        r_abs = abs(r)
        if r_abs == 1:
            return "--"
        elif r_abs > 0.9:
            return "Very High"
        elif r_abs > 0.7:
            return "High"
        elif r_abs > 0.5:
            return "Moderate"
        elif r_abs < 0.3:
            return "Low"
        elif r_abs < 0.1:
            return "Insignificant"
        else:
            return "Moderate"
