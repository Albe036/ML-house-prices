import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import os

df = pd.read_csv("./data/raw/train.csv")


class DataFrame_handling:
    def __init__(self, dataFrame, alpha=0.05):
        self.dataFrame = dataFrame
        self.alpha = alpha

    def _filter_features(self, type="numeric", custom_features=[]):
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

    def _config_output(self, res, onlyTrue=False):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        if onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df


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
# r > 0.7 → high (redundancia)
# r > 0.5 → moderate
# r < 0.3 → low
# ****************************************************************************************
class Features_correlations_pearson(DataFrame_handling):
    def pearson_correlation(self, featureAnalyze=""):
        # H_0 = No hay correlación entre featureAnalyze y las demás características numéricas
        # H_1 = Existe correlación entre featureAnalyze y al menos una de las demás características numéricas
        numeric_features = super()._filter_features(type="numeric")
        if featureAnalyze not in numeric_features.columns.tolist():
            raise ValueError(
                "feature must be numeric and present in the DataFrame"
            )
        cols = numeric_features.columns.tolist().remove(featureAnalyze)
        res = []
        for col in numeric_features:
            pearson_stat, p_value = pearsonr(
                numeric_features[featureAnalyze], numeric_features[col]
            )
            res.append(
                {
                    "feature": col,
                    "pearson_stat": pearson_stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                }
            )
        return super()._config_output(res)

    def pearson_correlation_one_to_one(self, featureAnalyze, featureCorrelation, with_plot=True):
        # H_0: No hay correlación entre featureAnalyze y featureCorrelation
        # H_1: Existe correlación entre featureAnalyze y featureCorrelation
        numeric_features = super()._filter_features(type="numeric")
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
                    "pearson_stat": pearson_stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                }
            ]
        )

    def calculate_pearson_matrix(self, custom_features=[], interpret=False):
        numeric_features = super()._filter_features(
            type="numeric", custom_features=custom_features
        )
        matrix = numeric_features.corr(method="pearson").round(2)
        if interpret:
            return matrix.apply(lambda x: x.apply(self.__interpret_pearson_value))
        return matrix
        #sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    def __interpret_pearson_value(self, r):
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
# ****************************************************************************************
# Spearman Correlation 
# El coeficiente de correlación de Spearman (ρ - rho) mide la relación monótona entre dos variables.
#  - Monótona significa que cuando una variable aumenta, la otra siempre aumenta o siempre disminuye,
#    pero no necesariamente a ritmo constante.
#  - Es un método no paramétrico porque trabaja con los rangos (orden) de los valores, no con los valores brutos.
# -------------------------------------------------
# +1: Correlación monótona perfecta positiva (cuando una sube, la otra siempre sube)
# -1: Correlación monótona perfecta negativa (cuando una sube, la otra siempre baja)
#  0: Ausencia de relación monótona (no significa independencia, como veremos)
# -------------------------------------------------

# ****************************************************************************************
class Features_correlations_spearman(DataFrame_handling):
    def spearman_correlation(self, featureAnalyze):
        # H_0 = No hay correlación entre featureAnalyze y las demás características numéricas
        # H_1 = Existe correlación entre featureAnalyze y al menos una de las demás características numéricas
        numeric_features = super()._filter_features(type="numeric")
        if featureAnalyze not in numeric_features.columns.tolist():
            raise ValueError(
                "feature must be numeric and present in the DataFrame"
            )
        cols = numeric_features.columns.tolist().remove(featureAnalyze)
        res = []
        for col in numeric_features:
            spearman_stat, p_value = spearmanr(
                numeric_features[featureAnalyze], numeric_features[col]
            )
            res.append(
                {
                    "feature": col,
                    "spearman_stat": spearman_stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                }
            )
        return super()._config_output(res)

    def spearman_correlation_one_to_one(self, featureAnalyze, featureCorrelation, with_plot=True):
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
    def calculate_pearson_matrix(self, custom_features=[], interpret=False):
        numeric_features = super()._filter_features(
            type="numeric", custom_features=custom_features
        )
        matrix = numeric_features.corr(method="spearman").round(2)
        if interpret:
            return matrix.apply(lambda x: x.apply(self.__interpret_pearson_value))
        return matrix
        #sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    def __interpret_pearson_value(self, r):
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