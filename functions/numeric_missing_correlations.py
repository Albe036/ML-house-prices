import numpy as np
import pandas as pd
from .missingHandling import MissingHandling
from scipy.stats import mannwhitneyu, ks_2samp, permutation_test, ttest_ind

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
Kolmogorov-Smirnov Test:
Comparación de distribuciones entre grupos con datos no paramétricos
1. Combina todos los datos
2. Calcula la función de distribución empírica (ECDF) para cada grupo
3. Calcula la estadística D de Kolmogorov-Smirnov, que es la máxima diferencia entre las ECDFs
4. Calcula el valor P
--------------------------------------------------------------------
Permutation test:
- Compara las medias de dos grupos independientes mediante permutaciones.
- No requiere que los datos sean normales.
- Sensible a la homogeneidad de varianzas.
--------------------------------------------------------------------
T-Student test:
- Compara las medias de dos grupos independientes.
- No requiere que los datos sean normales si los tamaños de muestra son grandes.
- Sensible a la homogeneidad de varianzas.
--------------------------------------------------------------------
"""


class NumericMissingCorrelations(MissingHandling):
    def mannWhitneyU(
        self, featureMissingValues=None, featuresReference=[], onlyTrue=True
    ):
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_types_features(
            custom_features=featuresReference, type_features="numerical"
        )
        for col in cols:
            missing, present, GREAT_ENOUGH = self._filter_missing_and_present(
                missing_bin, col
            )
            if GREAT_ENOUGH:
                stat, p_value = mannwhitneyu(missing, present, alternative="two-sided")
                res.append(
                    {
                        "feature": col,
                        "stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": (p_value < self.alpha),
                    }
                )
        return super()._config_output(res, onlyTrue=onlyTrue)

    def kolmogorovSmirnov(
        self, featureMissingValues=None, featuresReference=[], onlyTrue=True
    ):
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_types_features(
            custom_features=featuresReference, type_features="numerical"
        )
        for col in cols:
            missing, present, GREAT_ENOUGH = self._filter_missing_and_present(
                missing_bin, col
            )
            if GREAT_ENOUGH:
                stat, p_value = ks_2samp(missing, present, alternative="two-sided")
                res.append(
                    {
                        "feature": col,
                        "stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": (p_value < self.alpha),
                    }
                )
        return super()._config_output(res, onlyTrue=onlyTrue)

    @staticmethod
    def differenceMean(x, y):
        return np.mean(x) - np.mean(y)

    def permutationTest(
        self,
        featureMissingValues=None,
        featuresReference=[],
        iterations=100,
        onlyTrue=True,
    ):
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_types_features(
            custom_features=featuresReference, type_features="numerical"
        )
        for col in cols:
            missing, present, GREAT_ENOUGH = self._filter_missing_and_present(
                missing_bin, col
            )
            if GREAT_ENOUGH:
                res_permutation = permutation_test(
                    data=(missing, present),
                    statistic=NumericMissingCorrelations.differenceMean,
                    permutation_type="independent",
                    alternative="two-sided",
                    random_state=42,
                    n_resamples=iterations,
                )
                stat, p_value = res_permutation.statistic, res_permutation.pvalue
                res.append(
                    {
                        "feature": col,
                        "stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": (p_value < self.alpha),
                    }
                )
        return super()._config_output(res, onlyTrue=onlyTrue)

    def tStudent(self, featureMissingValues=None, featuresReference=[], onlyTrue=True):
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_types_features(
            custom_features=featuresReference, type_features="numerical"
        )
        for col in cols:
            missing, present, GREAT_ENOUGH = self._filter_missing_and_present(
                missing_bin, col
            )
            if GREAT_ENOUGH:
                stat, p_value = ttest_ind(missing, present, alternative="two-sided")
                res.append(
                    {
                        "feature": col,
                        "stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": (p_value < self.alpha),
                    }
                )
        return super()._config_output(res, onlyTrue=onlyTrue)
