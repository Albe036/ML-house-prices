import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions.correlations.FeaturesHandling import FeaturesHandling
from scipy.stats import chi2_contingency, association

# ****************************************************************************************
# Cramer's V:
# Cramér's V mide la fuerza de asociación entre dos variables categóricas.
# -------------------------------------------------
# Se basa en la prueba chi-cuadrado (χ²)
# Valores de 0 a 1 (NO negativos)
# 0: Independencia total (no hay relación)
# 1: Dependencia perfecta (una variable determina completamente a la otra)
# -------------------------------------------------
# ****************************************************************************************
class CramersVCorrelations(FeaturesHandling):
    def all_features(self, featureAnalyze="", onlyTrue=True, desc=False):
        categorical_cols = super()._filter_features(type_features="categorical")
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
    
    def one_to_one(
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
    def create_matrix(self, custom_features=[], interpret=False):
        categorical_features = super()._filter_features(
            type="categorical", custom_features=custom_features
        ) 
        matrix = categorical_features.corr(method="cramer").round(2)       
        if interpret:
            return matrix.apply(lambda x: x.apply(self._interpret_cramers_v_value))
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