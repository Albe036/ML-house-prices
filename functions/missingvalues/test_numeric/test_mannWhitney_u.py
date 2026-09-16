from functions.missingvalues.missingHandling import (
    MissingHandling,
    Interpretation_result_test,
    Effect_size_methods,
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
Spearman's rank correlation test:
- Spearman es una medida de correlación no paramétrica que mide la relación monótona
  entre dos variables. Es decir, si cuando una variable aumenta, la otra tiende a aumentar
  (o disminuir) de forma consistente.
- No requiere que los datos sean normales ni que la relación sea lineal.
Interpretabilidad abs(rho): Mide la fuerza y dirección de la relación monótona. Va de -1 a +1.
    0.0 <= rho < 0.1	Prácticamente nula	Correlación prácticamente irrelevante
    0.1 <= rho < 0.3	Muy débil	Correlación prácticamente irrelevante
    0.3 ≤ rho < 0.5	Débil	Correlación detectable pero pequeña
    0.5 ≤ rho < 0.7	Moderada	Correlación claramente perceptible
    rho ≥ 0.7	Fuerte	Correlación muy fuerte y relevante
Dirrecion (Correlacion):
    rho < 0       Negativa: La relación monótona es decreciente
    rho = 0       Sin correlación
    rho > 0       Positiva: La relación monótona es creciente
--------------------------------------------------------------------
"""


class MannWhitneyU(MissingHandling, Interpretation_result_test, Effect_size_methods):
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
                stat, p_value = mannwhitneyu(present, missing, alternative="two-sided")
                cohen_s = super()._calc_cohen_s(missing, present)
                rho, p_value_rho = super()._calc_spearman(
                    missing_feature_M=missing_M, reference_feature=col
                )
                values = {
                    "reference_feature": col,
                    "stat": stat,
                    "p_value": p_value,
                    "cohen_s": cohen_s,
                    "rho": rho,
                    "evidence_MAR": (p_value < self.alpha),
                }
                if desc:
                    values["p_value_interpretation"] = self._interpretate_p_value(
                        p_value
                    )
                    values["cohen_s_interpretation"] = self._interpret_cohen_s(cohen_s)
                    values["rho_interpretation"] = self._interpret_spearman(rho)
                    values["p_value_rho"] = p_value_rho
                res.append(values)
        return super()._config_output(res, desc=desc, onlyTrue=onlyTrue)
