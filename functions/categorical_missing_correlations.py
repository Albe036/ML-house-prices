from .missingHandling import MissingHandling
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


class CategoricalMissingCorrelations(MissingHandling):
    @staticmethod
    def _make_hashable(value):
        if isinstance(value, np.ndarray):
            return tuple(
                CategoricalMissingCorrelations._make_hashable(item)
                for item in value.tolist()
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                CategoricalMissingCorrelations._make_hashable(item)
                for item in value
            )
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (
                        key,
                        CategoricalMissingCorrelations._make_hashable(item),
                    )
                    for key, item in value.items()
                )
            )
        if isinstance(value, (set, frozenset)):
            return tuple(sorted(map(repr, value)))
        return value

    def applyTest(
        self,
        testFunc=None,
        featureMissingValues=None,
        featuresReference=None,
        onlyTrue=True,
        **kwargs,
    ):
        if featureMissingValues is None:
            raise ValueError("featureMissingValues must be specified")

        testFuncName = kwargs.pop("testFuncName", "")
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)

        cols = super()._filter_features(
            custom_features=featuresReference,
            typesFeatures=["object", "category", "string"],
        )

        for col in cols:
            feature = self.dataFrame[col].map(self._make_hashable)

            contingencyTable = pd.crosstab(missing_bin, feature)

            if contingencyTable.shape[1] < 2:
                continue

            if testFuncName == "fisher_exact" and contingencyTable.shape != (2, 2):
                continue

            stat, p_value = testFunc(contingencyTable.to_numpy(), **kwargs)

            res.append(
                {
                    "feature": col,
                    "stat": float(stat),
                    "p_value": float(p_value),
                    "evidence_MAR": bool(p_value < self.alpha),
                }
            )

        return super()._config_output(res, onlyTrue=onlyTrue)

    def chi2(self, featureMissingValues=None, featuresReference=None, onlyTrue=True):
        def chi2_test(contingencyTable):
            result = chi2_contingency(contingencyTable)
            return result.statistic, result.pvalue

        return self.applyTest(
            testFunc=chi2_test,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
        )

    def fisherExact(
        self, featureMissingValues=None, featuresReference=None, onlyTrue=True
    ):
        return self.applyTest(
            testFunc=fisher_exact,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
            testFuncName="fisher_exact",
        )

    def g_test(self, featureMissingValues=None, featuresReference=None, onlyTrue=True):
        def g_test_fun(contingencyTable):
            result = chi2_contingency(
                contingencyTable,
                lambda_="log-likelihood",
            )
            return result.statistic, result.pvalue

        return self.applyTest(
            testFunc=g_test_fun,
            featureMissingValues=featureMissingValues,
            featuresReference=featuresReference,
            onlyTrue=onlyTrue,
        )