
import pandas as pd
import numpy as np
from IPython.display import display

def createFeaturesMissingValuesList(df):
    df_train = df.copy()
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