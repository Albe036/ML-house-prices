import pandas as pd
import numpy as np
from IPython.display import display
from .missingHandling import MissingHandling

class UtilsMethods(MissingHandling):
    
    def createFeaturesMissingValuesList(self):
        df_train = self.dataFrame.copy()
        missing_data = df_train.isnull().sum()
        missing_data = pd.DataFrame(
            missing_data[missing_data > 0], columns=["missing counts"]
        )
        missing_data["percentage(%)"] = np.round(
            missing_data["missing counts"] / df_train.shape[0] * 100, 2
        )
        missing_data['type'] = df_train[missing_data.index].dtypes
        missing_data = missing_data.sort_values(by="percentage(%)", ascending=False)
        display(missing_data)
        return missing_data.index

    def addFeatureNormalDistribution(self, custom_name="normal_distribution"):
        normalSerie = np.random.randn(self.dataFrame.shape[0])
        self.dataFrame[custom_name] = normalSerie
        return self.dataFrame[custom_name]