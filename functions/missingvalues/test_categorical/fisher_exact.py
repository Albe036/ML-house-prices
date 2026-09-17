import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import association
from functions.missingvalues.missing_handling import MissingHandling

"""
--------------------------------------------------------------------
Fisher Exact Test: Es un test de independencia utilizado para tablas de contingencia 2x2.
Determina si existe una asociación significativa entre dos variables categóricas.
--------------------------------------------------------------------
"""
class FisherExactTest(MissingHandling):
    # def __init__(self, dataFrame, alpha=0.05):
    def all_features(
        self, custom_features=[], missing_feature="", desc=False, onlyTrue=True
    ):
        res = []
        # create feature Missing_M
        missing_M = super().create_missing_feature(missing_feature)
        # get categorical features
        cols = super()._create_list_features_types(
            type_features="categorical", custom_features=custom_features
        )
        for col in cols:
            contingency, NOT_SHORTAGE_VARIANCE = self._create_contingency_table(
                col, missing_M
            )
            NOT_2X2 = contingency.shape != (2, 2)
            if NOT_SHORTAGE_VARIANCE or NOT_2X2:
                continue
            oddstratio, p_value = fisher_exact(contingency.values)
            
    def _create_contingency_table(self, col, missing_M):
        contingency = pd.crosstab(self.dataFrame[col], self.dataFrame[missing_M])
        contingency.columns = ["present", "missing"]
        NOT_SHORTAGE_VARIANCE = contingency.shape[1] < 2
        return contingency, NOT_SHORTAGE_VARIANCE