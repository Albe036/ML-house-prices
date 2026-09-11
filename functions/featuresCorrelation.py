import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, chi2_contingency, association
import os

df = pd.read_csv("./data/raw/train.csv")

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
        # Creacion de la tabla de contingencia
        contingency_table = pd.crosstab(
            categorical_cols[featureAnalyze], categorical_cols[featureCorrelation]
        )
        # Prueba de chi-cuadrado (χ²) para la tabla de contingencia
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        # prueba de asociación usando Cramér's V
        cramers_v = association(contingency_table, method="cramer")
        if with_plot:
            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
            plt.title(f"Cramér's V: {cramers_v:.2f}")
            plt.show()

        return super()._config_output(
            {
                "featureAnalyze": featureAnalyze,
                "featureCorrelation": featureCorrelation,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "expected": expected,
                "cramers_v": cramers_v,
            }
        )
    def calculate_cramers_v_matrix(self, custom_features=[], interpret=False):
        categorical_features = super()._filter_features(
            type="categorical", custom_features=custom_features
        ) 
        matrix = categorical_features.corr(method="cramer").round(2)       
        if interpret:
            return matrix.apply(lambda: x.apply(self._interpret_cramers_v_value))
        return matrix
    
    def _interpret_cramers_v_value(self, r):
        if r == 0:
            return 'null'
        elif r <= 0.1:
            return 'very weak'
        elif r <= 0.3:
            return 'weak'
        elif r <= 0.5:
            return 'moderate'
        elif r <= 0.7:
            return 'strong'
        else:
            return 'very strong'

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
