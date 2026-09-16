import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from functions.missingvalues.missing_handling import MissingHandling

"""
--------------------------------------------------------------------
chi2: Chi-Cuadrado es un test de independencia que determina si existe relación entre dos variables categóricas.
1. crea tabla de contingencia
2. calcula frecuencias esperadas
3. compara frecuencias observadas con esperadas
4. calcula el valor p y el estadístico chi2
--------------------------------------------------------------------
Cramér's V: Es una medida de asociación entre dos variables categóricas basada en el estadístico chi2.
Su valor varía entre 0 y 1, donde 0 indica independencia y 1 indica asociación perfecta.
--------------------------------------------------------------------
Residuos Estandarizados: 
Son una medida que te dice cuánto se desvía cada celda de la tabla de contingencia
de lo que se esperaría si NO hubiera relación entre las variables.
--------------------------------------------------------------------
"""


class Chi2Test(MissingHandling):
    # def __init__(self, dataFrame, alpha=0.05):
    def all_features(
        self, custom_features=[], missing_feature="", desc=False, onlyTrue=True
    ):
        res = []
        # Create feature Missing_M
        missing_M = super().create_missing_feature(missing_feature)
        # get categorical features
        cols = super()._create_list_features_types(
            type_features="categorical", custom_features=custom_features
        )
        for col in cols:
            contingency, NOT_SHORTAGE_VARIANCE = self._create_contingency_table(
                col, missing_M
            )
            if NOT_SHORTAGE_VARIANCE:
                continue
            chi2, p_value, dof, expected = chi2_contingency(contingency.values)
            # apply cramersv
            cramers_v = association(contingency.values, method="cramer")
            # standardized residues
            missing_residues, missing_proportion = self.standardized_residues(
                contingency,
                expected
            )
            values = {
                "feature": col,
                "chi2": chi2,
                "p_value": p_value,
                "cramers_v": cramers_v,
                "missing_residues": missing_residues.tolist(),
                "missing_proportion": missing_proportion,
                "evidence_MAR": p_value < self.alpha,
            }
            if desc:
                values["p_value_inter"] = self._interpret_p_value(p_value)
                values["cramers_v_inter"] = self._interpret_cramers_v(cramers_v)
            res.append(values)
        return super()._config_output(res, desc=desc, onlyTrue=onlyTrue)

    def _create_contingency_table(self, col, missing_M):
        contingency = pd.crosstab(self.dataFrame[col], self.dataFrame[missing_M])
        contingency.columns = ["present", "missing"]
        NOT_SHORTAGE_VARIANCE = contingency.shape[1] < 2
        return contingency, NOT_SHORTAGE_VARIANCE

    def standardized_residues(self, contingency, expected):
        n = contingency.sum().sum()
        row_props = contingency.sum(axis=1) / n
        col_props = contingency.sum(axis=0) / n
        residues = (contingency.values - expected) / np.sqrt(
            expected * (1 - row_props.values[:, None]) * (1 - col_props.values[None, :])
        )
        residues = np.nan_to_num(residues, nan=0.0, posinf=0.0, neginf=0.0)
        # Residuo faltante
        missing_residues = residues[:, 1]
        # proporcion missing por categorias
        missing_proportion = contingency["missing"] / contingency.sum(axis=1)
        return missing_residues, missing_proportion

    def _interpret_cramers_v(self, cramers_v):
        if cramers_v < 0.1:
            return "insignificante"
        elif cramers_v < 0.3:
            return "debil"
        elif cramers_v < 0.5:
            return "moderada"
        elif cramers_v < 0.7:
            return "fuerte"
        else:
            return "muy fuerte"
