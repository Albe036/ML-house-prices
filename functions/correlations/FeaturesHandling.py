import pandas as pd
import numpy as np


# ****************************************************************************************
# FeaturesHandling class for handling and filtering features in a DataFrame
# Method One: Filter features:
#   - type_feature: numeric or categorical
#   - custom_features: list of specific features to filter
# Method Two: Configure output:
#   - res: DataFrame with results
#   - onlyTrue: filter rows where evidence_MAR is True
# ****************************************************************************************
class FeaturesHandling:
    def __init__(self, dataFrame, alpha=0.05):
        self.dataFrame = dataFrame
        self.alpha = alpha

    def _filter_features(self, type_features="numeric", custom_features=[]):
        custom_features = [f for f in custom_features if f in self.dataFrame.columns]
        if len(custom_features) > 0:
            if type_features == "numeric":
                return (
                    self.dataFrame[custom_features]
                    .select_dtypes(include=[np.number])
                    .dropna()
                )
            else:
                return (
                    self.dataFrame[custom_features]
                    .select_dtypes(exclude=[np.number])
                    .dropna()
                )
        else:
            if type_features == "numeric":
                return self.dataFrame.select_dtypes(include=[np.number]).dropna()
            else:
                return self.dataFrame.select_dtypes(exclude=[np.number]).dropna()

    def _config_output(self, res, onlyTrue=False):
        res_df = pd.DataFrame(res).sort_values(by="p_value", ascending=False)
        res_df['p_value'] = res_df['p_value'].round(5)
        if onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df
