from functions.missingvalues.missingHandling import (
    MissingHandling,
    Interpretation_result_test,
    Effect_size_methods,
)
import numpy as np
from scipy.stats import ttest_ind, pearsonr

""" 
USADA CUANDO LOS DATOS TIENEN UNA DISTRIBUCION NORMAL
--------------------------------------------------------------------
T-Student test:
- Compara las medias de dos grupos independientes.
- No requiere que los datos sean normales si los tamaños de muestra son grandes.
- Sensible a la homogeneidad de varianzas.
--------------------------------------------------------------------
Cohen's d test:
Cohen's d es una medida de tamaño del efecto que cuantifica cuánta diferencia
hay entre las medias de dos grupos, expresada en unidades de desviación estándar.
Interpretabilidad abs(d):
    d < 0.2	Muy pequeño	Diferencia prácticamente irrelevante
    0.2 ≤ d < 0.5	Pequeño	Diferencia detectable pero pequeña
    0.5 ≤ d < 0.8	Moderado	Diferencia claramente perceptible
    d ≥ 0.8	Grande	Diferencia muy grande y relevante
Dirrecion:
    d < 0       Negativa: El grupo 1 tiene media menor que el grupo 2
    d = 0       Sin diferencia
    d > 0       Positiva: El grupo 1 tiene media mayor que el grupo 2
--------------------------------------------------------------------
Pearson correlation:
Mide la fuerza y dirección de la relación lineal entre dos variables.
El valor de correlación r varía entre -1 y 1:
    r = 1   Correlación positiva perfecta
    r = -1  Correlación negativa perfecta
    r = 0   Sin correlación lineal
--------------------------------------------------------------------
"""


class TStudent(MissingHandling):
    # def __init__(self, dataFrame, alpha=0.05):
    def all_features(
        self, custom_features=[], missing_feature="", desc=False, onlyTrue=True
    ):
        res = []
        # Create feature missing_M
        missing_M = super()._define_missing_feature(missing_feature=missing_feature)
        # Get numerical features
        cols = super()._create_list_features_types(
            type_features="numerical", custom_features=custom_features
        )
        # Define present y missing for feature with missing values
        for col in cols:
            missing, present, GREAT_ENOUGH = self._split_groups(
                missing_feature=missing_feature,
                reference_feature=col,
            )
            if GREAT_ENOUGH:
                stat, p_value = ttest_ind(missing, present)
                cohen_s = super()._calc_cohen_s(missing, present)
                r, p_value_r = super()._calc_pearson(
                    missing_feature_M=missing_M, reference_feature=col
                )
                values = {
                    "reference_feature": col,
                    "stat": stat,
                    "p_value": p_value,
                    "cohen_s": cohen_s,
                    "r": r,
                    "p_value_r": p_value_r,
                    "evidence_MAR": (p_value < self.alpha),
                }
                if desc:
                    values["p_value_interpretation"] = self._interpretate_p_value(
                        p_value
                    )
                    values["cohen_s_interpretation"] = self._interpret_cohen_s(cohen_s)
                    values["r_interpretation"] = self._interpret_pearson(r)
                    values["p_value_r"] = p_value_r
                res.append(values)
        return super()._config_output(res, desc=desc, onlyTrue=onlyTrue)
