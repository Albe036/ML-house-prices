from functions.missingvalues.missingHandling import MissingHandling
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
                cohen_s = self._calc_cohen_s(missing, present)
                r, p_value_r = self._calc_pearson(
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

    def _calc_cohen_s(self, missing_values, present_values):
        mean_missing, mean_present = missing_values.mean(), present_values.mean()
        std1_missing, std1_present = missing_values.std(ddof=1), present_values.std(
            ddof=1
        )
        len_missing, len_present = len(missing_values), len(present_values)

        # Desvio estandar pooled
        pooled_std = np.sqrt(
            ((len_missing - 1) * std1_missing**2 + (len_present - 1) * std1_present**2)
            / (len_missing + len_present - 2)
        )

        # Cohen's d
        cohen_d = (
            ((mean_missing - mean_present) / pooled_std)
            if np.isfinite(pooled_std) and pooled_std != 0
            else 0
        )
        return cohen_d

    def _interpret_cohen_s(self, cohen_s):
        direction = (
            "Negativa"
            if cohen_s < 0
            else "Positiva" if cohen_s > 0 else "Sin diferencia"
        )
        if abs(cohen_s) < 0.2:
            return f"Muy pequeño ({direction})"
        elif 0.2 <= abs(cohen_s) < 0.5:
            return f"Pequeño ({direction})"
        elif 0.5 <= abs(cohen_s) < 0.8:
            return f"Moderado ({direction})"
        else:
            return f"Grande ({direction})"

    def _calc_pearson(self, missing_feature_M, reference_feature):
        data = self.dataFrame[[missing_feature_M, reference_feature]].dropna()
        if len(data) < 3:
            return 0, 1  # Not enough data to calculate Pearson correlation
        r, p_value_r = pearsonr(data[missing_feature_M], data[reference_feature])
        return r, p_value_r

    def _interpret_pearson(self, r):
        direction = (
            "Negativa"
            if r < 0
            else "Positiva" if r > 0 else "Sin correlación"
        )
        if abs(r) < 0.1:
            return f"Insignificante ({direction})"
        elif 0.1 <= abs(r) < 0.3:
            return f"Débil ({direction})"
        elif 0.3 <= abs(r) < 0.5:
            return f"Moderada ({direction})"
        elif 0.5 <= abs(r) < 0.7:
            return f"Fuerte ({direction})"
        else:
            return f"Muy fuerte ({direction})"
