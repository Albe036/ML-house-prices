import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, chi2_contingency, association
import os

df = pd.read_csv("./data/raw/train.csv")


class FeaturesHandling:
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
class Features_correlations_pearson(FeaturesHandling):
    def pearson_correlation(self, featureAnalyze=""):
        # H_0 = No hay correlación entre featureAnalyze y las demás características numéricas
        # H_1 = Existe correlación entre featureAnalyze y al menos una de las demás características numéricas
        numeric_features = super()._filter_features(type="numeric")
        cols = numeric_features.columns.tolist()
        if featureAnalyze not in cols:
            raise ValueError("feature must be numeric and present in the DataFrame")
        cols = cols.remove(featureAnalyze)
        res = []
        for col in cols:
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

    def pearson_correlation_one_to_one(
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
        # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

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
# Coeficiente de correlacion:
# +1: Correlación monótona perfecta positiva (cuando una sube, la otra siempre sube)
# -1: Correlación monótona perfecta negativa (cuando una sube, la otra siempre baja)
#  0: Ausencia de relación monótona (no significa independencia, como veremos)
# -------------------------------------------------
# Cuanto usar spearman:
# - Relaciones no lineales pero monótonas:
# - Datos con outliers extremos: Al usar rangos, los outliers pierden su efecto distorsionador.
# - Variables ordinales: Cuando tienes escalas como "bajo", "medio", "alto" codificadas como 1, 2, 3.
# - Los datos NO siguen una distribución normal: Spearman no asume normalidad.
# ****************************************************************************************
class Features_correlations_spearman(FeaturesHandling):
    def spearman_correlation(self, featureAnalyze):
        # H_0 = No hay correlación entre featureAnalyze y las demás características numéricas
        # H_1 = Existe correlación entre featureAnalyze y al menos una de las demás características numéricas
        numeric_features = super()._filter_features(type="numeric")
        cols = numeric_features.columns.tolist()
        if featureAnalyze not in cols:
            raise ValueError("feature must be numeric and present in the DataFrame")
        cols = cols.remove(featureAnalyze)
        res = []
        for col in cols:
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

    def spearman_correlation_one_to_one(
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

    def calculate_spearman_matrix(self, custom_features=[], interpret=False):
        numeric_features = super()._filter_features(
            type="numeric", custom_features=custom_features
        )
        matrix = numeric_features.corr(method="spearman").round(2)
        if interpret:
            return matrix.apply(lambda x: x.apply(self.__interpret_spearman_value))
        return matrix
        # sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

    def __interpret_spearman_value(self, r):
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


# ****************************************************************************************
# Cramer's V:
# Cramér's V mide la fuerza de asociación entre dos variables categóricas.
# -------------------------------------------------
# Se basa en la prueba chi-cuadrado (χ²)
# Valores de 0 a 1 (NO negativos)
# 0: Independencia total (no hay relación)
# 1: Dependencia perfecta (una variable determina completamente a la otra)
# ****************************************************************************************
class Features_correlations_cramers_v:
    def calculate_cramers_v(self, featureAnalyze):
        categorical_cols = super()._filter_features(type="categorical")
        cols = categorical_cols.columns.tolist()
        if featureAnalyze not in cols:
            raise ValueError(
                "featureAnalyze must be categorical and present in the DataFrame"
            )
        res = []
        cols.remove(featureAnalyze)
        for col in cols:
            # Tabla de contingencia (frecuencias observadas)
            contingeny_table = pd.crosstab(
                categorical_cols[featureAnalyze], categorical_cols[col]
            )
            # Prueba de chi-cuadrado (χ²)
            chi2, p, dof, expected = chi2_contingency(contingeny_table)
            # Numero total de observaciones
            n = contingeny_table.sum().sum()

            # Calcular Cramér's V
            cramers_v = association(contingeny_table, method="cramer")
            res.append(
                {
                    "featureAnalyze": featureAnalyze,
                    "featureCorrelation": col,
                    "chi2": chi2,
                    "p_value": p,
                    "dof": dof,
                    "expected": expected,
                    "n": n,
                    "cramers_v": cramers_v,
                }
            )
        return super()._config_output(res)

    def cramers_correlation_one_to_one(
        self, featureAnalyze, featureCorrelation, with_plot=False
    ):
        # H_0: No hay asociación entre featureAnalyze y featureCorrelation (Cramér's V = 0)
        # H_1: Existe asociación entre featureAnalyze y featureCorrelation (Cramér's V > 0)
        categorical_cols = super()._filter_features(type="categorical")
        cols = categorical_cols.columns.tolist()
        if featureAnalyze not in cols or featureCorrelation not in cols:
            raise ValueError(
                "Both featureAnalyze and featureCorrelation must be categorical and present in the DataFrame"
            )
        #Creacion de la tabla de contingencia
        contingency_table = pd.crosstab(
            categorical_cols[featureAnalyze], categorical_cols[featureCorrelation]
        )
        # Prueba de chi-cuadrado (χ²) para la tabla de contingencia
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        cramers_v = association(contingency_table, method="cramer")
        if with_plot:
            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
            plt.title(f"Cramér's V: {cramers_v:.2f}")
            plt.show()

        return cramers_v


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
