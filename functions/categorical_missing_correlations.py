from .missingHandling import MissingHandling
import pandas as pd
from scipy.stats import chi2_contingency


class CategoricalMissingCorrelations(MissingHandling):
    def chi2(self, featureMissingValues=None, featuresReference=[], onlyTrue=True):
        res = []
        missing_bin = self.dataFrame[featureMissingValues].isna().astype(int)
        cols = super()._filter_types_features(
            custom_features=featuresReference, type_features="categorical"
        )
        for col in cols:
            contingency, NO_SHORTAGE_VARIANCE = (
                CategoricalMissingCorrelations._create_contingency(
                    missing_bin, self.dataFrame[col]
                )
            )
            if NO_SHORTAGE_VARIANCE:
                continue
            stat, p_value, dof, expected = chi2_contingency(contingency.values)
            res.append(
                {
                    "feature": col,
                    "stat": stat,
                    "p_value": p_value,
                    "dof": dof,
                    "expected": expected,
                    "evidence_MAR": (p_value < self.alpha),
                }
            )
        return super()._config_output(res, onlyTrue=onlyTrue)

    @staticmethod
    def _create_contingency(featureMissingValues=None, currentFeatureReference=None):
        con = pd.crosstab(featureMissingValues, currentFeatureReference)
        NO_SHORTAGE_VARIANCE = con.shape[1] < 2
        return con, NO_SHORTAGE_VARIANCE
