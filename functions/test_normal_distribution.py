import numpy as np
import pandas as pd
from .missingHandling import MissingHandling
from scipy.stats import anderson, kstest, normaltest, shapiro, jarque_bera
from statsmodels.stats.diagnostic import lilliefors



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
    def applyTest(self, testFunc=None, custom_features=[], **kwargs):
        res = []
        cols = self._filter_features(custom_features=custom_features)
        for col in cols:
            stat, p_value = testFunc(self.dataFrame[col].dropna(), **kwargs)
            res.append(
                {
                    "name_feature": col,
                    "stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return pd.DataFrame(res)

    def shapiro_wilk_test(self, custom_features=[]):
        return self.applyTest(testFunc=shapiro, custom_features=custom_features)

    def kolmogorov_smirnov_test(self, custom_features=[]):
        def kstest_custom(data):
            mu, sigma = data.mean(), data.std()
            stat, p_value = kstest(data, "norm", args=(mu, sigma))
            return stat, p_value

        return self.applyTest(testFunc=kstest_custom, custom_features=custom_features)

    def lilliefors_test(self, custom_features=[]):
        return self.applyTest(
            testFunc=lilliefors, custom_features=custom_features, dist="norm"
        )

    def anderson_darling_test(self, custom_features=[]):
        def anderson_darling_custom(data):
            result = anderson(data.dropna(), dist="norm")
            stat = result.statistic
            crit_values = result.critical_values
            sig_levels = result.significance_level
            idx_05 = np.argmin(np.abs(np.array(sig_levels) - 5.0))
            evidence_non_normality = stat > crit_values[idx_05]
            return stat, crit_values, sig_levels, evidence_non_normality

        return self.applyTest(
            testFunc=anderson_darling_custom, custom_features=custom_features
        )

    def dagostino_pearson_test(self, custom_features=[]):
        return self.applyTest(testFunc=normaltest, custom_features=custom_features)

    def jarque_bera_test(self, custom_features=[]):
        return self.applyTest(testFunc=jarque_bera, custom_features=custom_features)

    def _config_output(self, res, byOrder="p_value"):
        for property in res:
            for key, value in property.items():
                if isinstance(value, float):
                    property[key] = round(value, 5)
        res_df = pd.DataFrame(res).sort_values(by=byOrder)
        return res_df.reset_index(drop=True)
