import numpy as np
import pandas as pd


class MissingHandling:
    def __init__(self, dataFrame, alpha=0.05):
        self.dataFrame = dataFrame
        self.alpha = alpha

    def _define_missing_feature(self, missing_feature=""):
        # Define new feature missing: True and present: False
        missing_feature_M = f"{missing_feature}_M"
        self.dataFrame[missing_feature_M] = self.dataFrame[missing_feature].isna()
        return missing_feature_M

    def _create_list_features_types(
        self, custom_features=[], type_features="numerical"
    ):
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

    def _split_groups(
        self,
        missing_feature="",
        reference_feature="",
    ):
        if missing_feature not in self.dataFrame.columns.tolist():
            raise ValueError(
                f"Missing feature '{missing_feature}' not found in the DataFrame."
            )

        # split missing and present groups based on the reference feature
        MIN_ABSOLUTE_GROUP_SIZE = 3
        missing = self.dataFrame.loc[
            self.dataFrame[missing_feature].isna(), reference_feature
        ].dropna()
        present = self.dataFrame.loc[
            self.dataFrame[missing_feature].notna(), reference_feature
        ].dropna()
        len_missing = len(missing)
        len_present = len(present)

        GREAT_ENOUGH = (
            len_missing >= MIN_ABSOLUTE_GROUP_SIZE
            and len_present >= MIN_ABSOLUTE_GROUP_SIZE
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
