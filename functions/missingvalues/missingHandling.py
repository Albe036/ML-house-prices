import numpy as np
import pandas as pd

class MissingHandling:
    def __init__(self, dataFrame, missingFeature="", alpha=0.05, onlyTrue = True):
        self.dataFrame = dataFrame
        self.missingFeature = missingFeature
        self.missingFeature_M = f"{missingFeature}_M"
        self.alpha = alpha
        self.onlyTrue = onlyTrue

    def _define_missing_feature(self):
        #Define new feature missing: True and present: False
        self.dataFrame[self.missingFeature_M] = self.dataFrame[self.missingFeature].isna()

    def _define_type_features(self, custom_features=[], type_features="numerical"):
        custom_features = [f for f in custom_features if f in self.dataFrame.columns.tolist()]
        if len(custom_features) == 0:
            custom_features = self.dataFrame.columns.tolist()
        if type_features == "numerical":
            cols = self.dataFrame[custom_features].select_dtypes(include=["number"]).columns.tolist()
        else:
            cols = self.dataFrame[custom_features].select_dtypes(exclude=["number"]).columns.tolist()
        if "Id" in cols:
            cols.remove("Id")
        return cols
        

    def _split_groups(self, reference_feature="", missing_rows=None, present_rows=None):
        if self.missingFeature not in self.dataFrame.columns.tolist():
            raise ValueError(f"Missing feature '{self.missingFeature}' not found in the DataFrame.")

        #split missing and present groups based on the reference feature
        MIN_ABSOLUTE_GROUP_SIZE = 3
        """ missing = self.dataFrame.loc[
            self.dataFrame[self.missingFeature].isna(), reference_feature
        ].dropna()
        present = self.dataFrame.loc[
            self.dataFrame[self.missingFeature].notna(), reference_feature
        ].dropna() """
        missing = self.dataFrame.loc[missing_rows, reference_feature].dropna()
        present = self.dataFrame.loc[present_rows, reference_feature].dropna()
        len_missing = len(missing)
        len_present = len(present)

        GREAT_ENOUGH = (len_missing >= MIN_ABSOLUTE_GROUP_SIZE and len_present >= MIN_ABSOLUTE_GROUP_SIZE)
        if not GREAT_ENOUGH:
            return None, None, 0, 0, False
        return missing, present, len_missing, len_present, GREAT_ENOUGH

    def _config_output(self, res):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        if self.onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df
