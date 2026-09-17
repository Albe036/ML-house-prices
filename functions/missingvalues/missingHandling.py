import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


class MissingHandling:

    METHOD_SIZE_SPEARMAN = "SPEARMAN"
    METHOD_SIZE_PEARSON = "PEARSON"
    METHOD_DIRECTION_COHENS = "COHENS"

    def __init__(self, dataFrame, alpha=0.05, maxGroup=3):
        self.dataFrame = dataFrame
        self.alpha = alpha
        self.MIN_ABSOLUTE_GROUP_SIZE = maxGroup

    def _filter_types_features(self, custom_features=[], type_features="numerical"):
        custom_features = [
            f for f in custom_features if f in self.dataFrame.columns.tolist()
        ]
        if len(custom_features) == 0:
            custom_features = self.dataFrame.columns.tolist()
        if type_features == "numerical":
            cols = (
                self.dataFrame[custom_features]
                .select_dtypes(include=["number"])
                .columns.tolist()
            )
        else:
            cols = (
                self.dataFrame[custom_features]
                .select_dtypes(exclude=["number"])
                .columns.tolist()
            )
        if "Id" in cols:
            cols.remove("Id")
        return cols

    def _filter_missing_and_present(self, missing_feature=None, reference_feature=""):
        # split missing and present groups based on the reference feature
        missing = self.dataFrame.loc[missing_feature == 1, reference_feature].dropna()
        present = self.dataFrame.loc[missing_feature == 0, reference_feature].dropna()
        len_missing = len(missing)
        len_present = len(present)

        GREAT_ENOUGH = (
            len_missing >= self.MIN_ABSOLUTE_GROUP_SIZE
            and len_present >= self.MIN_ABSOLUTE_GROUP_SIZE
        )
        if not GREAT_ENOUGH:
            return None, None, False
        return missing, present, GREAT_ENOUGH

    def _config_output(self, res, desc=False, onlyTrue=True):
        for property in res:
            for key, value in property.items():
                if isinstance(value, float):
                    property[key] = round(value, 5)
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        if onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
            res_df.drop(columns=["evidence_MAR"], inplace=True)
        return res_df.reset_index(drop=True)


class Methods_effect_size:
    METHOD_SIZE_EFFECT_COHEN_S = "cohen_s"

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

    def calc_size_effect(self, method, missing_values, present_values):
        if method == Methods_effect_size.METHOD_SIZE_EFFECT_COHEN_S:
            return (
                self._calc_cohen_s(self.missing_values, self.present_values),
                Methods_effect_size.METHOD_SIZE_EFFECT_COHEN_S,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Tamaños del efecto


class Effect_size_methods:
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

    # Direccion de la magnitud
    def _calc_spearman(self, missing_feature_M, reference_feature):
        data = self.dataFrame[[missing_feature_M, reference_feature]].dropna()
        if len(data) < 3:
            return 0.0, 1.0
        rho, p_value_rho = spearmanr(
            data[missing_feature_M],
            data[reference_feature],
        )
        return rho, p_value_rho

    def _calc_pearson(self, missing_feature_M, reference_feature):
        data = self.dataFrame[[missing_feature_M, reference_feature]].dropna()
        if len(data) < 3:
            return 0, 1  # Not enough data to calculate Pearson correlation
        r, p_value_r = pearsonr(data[missing_feature_M], data[reference_feature])
        return r, p_value_r


"""
Interpretation of test results for missing value analysis.
--------------------------------------------------------------------
p_value: Significance level of the test. A low p_value indicates strong evidence against the null hypothesis.
--------------------------------------------------------------------
cohen_s: Measure of effect size. Indicates the standardized difference between two means.
--------------------------------------------------------------------
rho: Spearman's rank correlation coefficient. Measures the strength and direction of the monotonic relationship between two variables.
--------------------------------------------------------------------
r: Pearson correlation coefficient. Measures the strength and direction of the linear relationship between two variables.
--------------------------------------------------------------------
"""


class Interpretation_result_test:
    # Interpretación del valor P
    def _interpretate_p_value(self, p_value):
        if p_value == 0:
            return "perfect!!"
        if p_value < 0.001:
            return "very high"
        if p_value >= 0.001 and p_value < 0.01:
            return "high"
        if p_value >= 0.01 and p_value < 0.05:
            return "moderate"
        if p_value >= 0.05 and p_value < 0.1:
            return "low"
        return "very low"

    def _interpret_spearman(self, rho):
        direction = (
            "Negativa" if rho < 0 else "Positiva" if rho > 0 else "Sin correlación"
        )

        if abs(rho) < 0.1:
            return f"Insignificante ({direction})"
        elif 0.1 <= abs(rho) < 0.3:
            return f"Débil ({direction})"
        elif 0.3 <= abs(rho) < 0.5:
            return f"Moderada ({direction})"
        elif 0.5 <= abs(rho) < 0.7:
            return f"Fuerte ({direction})"
        else:
            return f"Muy fuerte ({direction})"

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

    def _interpret_pearson(self, r):
        direction = "Negativa" if r < 0 else "Positiva" if r > 0 else "Sin correlación"
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
