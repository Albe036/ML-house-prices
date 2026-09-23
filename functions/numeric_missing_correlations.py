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
    def applyTest(
        self,
        testFunc=None,
        effectSizeAndDirection=None,
        featureMissingValues=None,
        featuresReference=None,
        onlyTrue=True,
        **kwargs
    ):
        if featureMissingValues is None:
            raise ValueError("featureMissingValues must be specified")
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_features(custom_features=featuresReference)
        for col in cols:
            missing = self.dataFrame.loc[missing_bin == 1, col].dropna()
            present = self.dataFrame.loc[missing_bin == 0, col].dropna()
            if (len(missing) < self.MIN_ABSOLUTE_GROUP_SIZE) or (
                len(present) < self.MIN_ABSOLUTE_GROUP_SIZE
            ):
                continue
            stat, p_value = testFunc(missing, present, **kwargs)
            values = {
                "feature": col,
                "stat": stat,
                "p_value": p_value,
                "evidence_MAR": (p_value < self.alpha),
            }
            if effectSizeAndDirection is not None:
                values = values | effectSizeAndDirection(stat, p_value, missing=missing, present=present)
            res.append(values)
        return pd.DataFrame(res)

    def mannWhitneyU(
        self, featureMissingValues=None, featuresReference=[], effectSizeAndDirection=None, onlyTrue=True
    ):
        return self.applyTest(
            mannwhitneyu,
            effectSizeAndDirection=effectSizeAndDirection,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
            alternative="two-sided",
        )

    def kolmogorovSmirnov(
        self, featureMissingValues=None, featuresReference=[], onlyTrue=True
    ):
        return self.applyTest(
            ks_2samp,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
            alternative="two-sided",
        )

    def permutationTest(
        self,
        featureMissingValues=None,
        featuresReference=[],
        iterations=1000,
        onlyTrue=True,
    ):
        def permutation_statistic(missing, present):
            res_permutation = permutation_test(
                data=(missing, present),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                permutation_type="independent",
                alternative="two-sided",
                random_state=42,
                n_resamples=iterations,
            )
            return res_permutation.statistic, res_permutation.pvalue

        return self.applyTest(
            permutation_statistic,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
        )

    def tStudent(self, featureMissingValues=None, featuresReference=[], onlyTrue=True):
        return self.applyTest(
            ttest_ind,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
            alternative="two-sided",
            equal_var=False,
        )
