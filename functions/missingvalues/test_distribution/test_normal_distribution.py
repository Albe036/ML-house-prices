import numpy as np
import pandas as pd
from scipy.stats import anderson, kstest, normaltest, shapiro, jarque_bera
from functions.missingvalues.missingHandling import MissingHandling


# ---------------------------------------------------------------------
# H0: La muestra proviene de una distribución normal
# ---------------------------------------------------------------------
# SHAPIRO-WILK TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística W de Shapiro-Wilk y el valor P
# 2. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n < 5000)
# Sensible a outliers y a la asimetría de la distribución
# ---------------------------------------------------------------------
# KOLMOGOROV-SMIRNOV TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística D de Kolmogorov-Smirnov y el valor P
# 2. Evalúa la significancia estadística comparando el valor P
#    con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n <= 5000) y distribuciones continuas
# ---------------------------------------------------------------------
# ANDERSON-DARLING TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística A de Anderson-Darling y los valores críticos
# 2. Evalúa la significancia estadística comparando la estadística A con los valores críticos
# 3. Si la estadística A es mayor que el valor crítico correspondiente al nivel de significancia (alpha), se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n >= 5000)
# ---------------------------------------------------------------------
# DAGOSTINO-PEARSON TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística D de D'Agostino-Pearson y el valor P
# 2. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n >= 20)
# ---------------------------------------------------------------------
# JARQUE-BERA TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística JB de Jarque-Bera y el valor P
# 2. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n >= 20)
# ---------------------------------------------------------------------
class NormalDistributionTest(MissingHandling):
    def shapiro_wilk_test(self, custom_features=[]):
        res = []
        list_cols = self._create_list_features_types(custom_features)
        for col in list_cols:
            stat, p_value = shapiro(self.dataFrame[col].dropna())
            res.append(
                {
                    "name_feature": col,
                    "shapiro_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self._config_output(res, onlyTrue=False, desc=False)

    def kolmogorov_smirnov_test(self, custom_features=[]):
        res = []
        list_cols = self._create_list_features_types(custom_features)
        for col in list_cols:
            data = self.dataFrame[col].dropna()
            mu, sigma = data.mean(), data.std()
            stat, p_value = kstest(data, "norm", args=(mu, sigma))
            res.append(
                {
                    "name_feature": col,
                    "kolmogorov_smirnov_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self._config_output(res, onlyTrue=False, desc=False)

    def anderson_darling_test(self, custom_features=[]):
        res = []
        list_cols = self._create_list_features_types(custom_features)
        for col in list_cols:
            result = anderson(self.dataFrame[col].dropna(), dist="norm")
            stat = result.statistic
            crit_values = result.critical_values
            sig_levels = result.significance_level
            idx_05 = idx_05 = np.argmin(np.abs(np.array(sig_levels) - 5.0)) # Nivel de significancia del 5%
            evidence_non_normality = stat > crit_values[idx_05]  # Typically using the 5% significance level
            res.append(
                {
                    "name_feature": col,
                    "anderson_darling_stat": stat,
                    "critical_values": crit_values,
                    "significance_levels": sig_levels,
                    "evidence_non_normality": evidence_non_normality,
                    "interpretacion": (
                        "No normal" if evidence_non_normality else "Normal"
                    ),
                }
            )
        return self._config_output(res, onlyTrue=False, desc=False, byOrder="anderson_darling_stat")

    def dagostino_pearson_test(self, custom_features=[]):
        res = []
        list_cols = self._create_list_features_types(custom_features)
        for col in list_cols:
            stat, p_value = normaltest(self.dataFrame[col].dropna())
            res.append(
                {
                    "name_feature": col,
                    "dagostino_pearson_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self._config_output(res, onlyTrue=False, desc=False)

    def jarque_bera_test(self, custom_features=[]):
        res = []
        list_cols = self._create_list_features_types(custom_features)
        for col in list_cols:
            stat, p_value = jarque_bera(self.dataFrame[col].dropna())
            res.append(
                {
                    "name_feature": col,
                    "jarque_bera_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self._config_output(res, onlyTrue=False, desc=False) 

    def _config_output(self, res, desc=False, onlyTrue=True, byOrder="p_value"):
            for property in res:
                for key, value in property.items():
                    if isinstance(value, float):
                        property[key] = round(value, 5)
            res_df = pd.DataFrame(res).sort_values(by=byOrder)
            if onlyTrue:
                res_df = res_df[res_df["evidence_non_normality"]]
                res_df.drop(columns=["evidence_non_normality"], inplace=True)
            return res_df.reset_index(drop=True)