from functions.missingvalues.missingHandling import MissingHandling
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
class MannWhitneyU(MissingHandling):
    # def __init__(self, dataFrame, alpha=0.05, onlyTrue=True):
    def all_features(self, custom_features=[], missing_feature="", desc=False):
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
                cohen_s = self.calc_cohen_s(missing, present)
                rho, p_value_rho = self.calc_spearman(
                    missing_feature_M=missing_M, reference_feature=col
                )
                res.append(
                    {
                        "reference_feature": col,
                        "stat": stat,
                        "p_value": p_value,
                        "cohen_s": cohen_s,
                        "cohen_s_interpretation": self._interpret_cohen_s(cohen_s),
                        "rho_interpretation": self._interpret_spearman(rho),
                        "rho": rho,
                        "p_value_rho": p_value_rho,
                        "evidence_MAR": (p_value < self.alpha),
                    }
                )
        return super()._config_output(res, desc=desc)

    def calc_cohen_s(self, missing_values, present_values):
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
        direction = "Negativa" if cohen_s < 0 else "Positiva" if cohen_s > 0 else "Sin diferencia"
        cohen_s = cohen_s / 100
        if abs(cohen_s) < 0.2:
            return f"Prácticamente nula ({direction})"
        elif 0.2 <= abs(cohen_s) < 0.5:
            return f"Débil ({direction})"
        elif 0.5 <= abs(cohen_s) < 0.7:
            return f"Moderada ({direction})"
        else:
            return f"Fuerte ({direction})"

    def calc_spearman(self, missing_feature_M, reference_feature):
        rho, p_value_rho = spearmanr(
            self.dataFrame[missing_feature_M],
            self.dataFrame[reference_feature],
        )
        return rho, p_value_rho

    def _interpret_spearman(self, rho):
        direction = "Negativa" if rho < 0 else "Positiva" if rho > 0 else "Sin correlación"

        if abs(rho) < 0.1:
            return f"Prácticamente nula ({direction})"
        elif 0.1 <= abs(rho) < 0.3:
            return f"Muy débil ({direction})"
        elif 0.3 <= abs(rho) < 0.5:
            return f"Débil ({direction})"
        elif 0.5 <= abs(rho) < 0.7:
            return f"Moderada ({direction})"
        else:
            return f"Fuerte ({direction})"
