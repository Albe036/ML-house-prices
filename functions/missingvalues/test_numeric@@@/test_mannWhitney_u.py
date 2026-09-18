from functions.missingvalues.missingHandling import (
    MissingHandling,
    Interpretation_result_test,
    Effect_size_methods,
    Methods_effect_size,
)
from scipy.stats import mannwhitneyu, spearmanr
import numpy as np

""" 
--------------------------------------------------------------------
Mann-Whitney U Test:
Comparación de distribuciones entre grupos con datos no paramétricos
1. Combina todos los datos
2. Asigna rangos a los datos combinados
3. Divide los datos en dos grupos: presentes y ausentes (por rangos)
4. Calcula la estadística U de Mann-Whitney para cada grupo y escoge el menor
5. Calcula el valor P
--------------------------------------------------------------------
"""


class MannWhitneyU(MissingHandling, Methods_effect_size):
    # def __init__(self, dataFrame, alpha=0.05):
    def all_features(
        self,
        custom_features=[],
        missing_feature="",
        method_size_effect=None,
        desc=False,
        onlyTrue=True,
    ):
        res = []
        # Create feature missing_M
        missing_binary_feature = self.dataFrame[missing_feature].isna().astype(int)
        # filter features
        cols = super()._filter_types_features(custom_features=custom_features)
        # Define present y missing for feature with missing values
        for col in cols:
            missing, present, GREAT_ENOUGH = self._filter_missing_and_present(missing_binary_feature, col)
            if GREAT_ENOUGH:
                stat, p_value = mannwhitneyu(missing, present, alternative="two-sided")
                values = {
                    "reference_feature": col,
                    "stat": stat,
                    "p_value": p_value,
                    "evidence_MAR": (p_value < self.alpha),
                }
                if method_size_effect:
                    name_method, size_effect = calc_size_effect(method_size_effect, missing, present)
                    values[name_method] = size_effect
                res.append(values)
        return super()._config_output(res, desc=desc, onlyTrue=onlyTrue)
