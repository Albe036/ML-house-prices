import pandas as pd
from pathlib import Path
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
useTrain = pd.read_csv(TRAIN_PATH)

X = useTrain.drop(columns=["SalePrice"])
y = useTrain["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# imputacion de datos faltantes
# Clase imputacion
class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.modeBsmtExposure = None
        self.modeElectrical = None
        self.mappingLotFrontage = None
        self.globalMeanLotFrontage = None
        self.FEATURES_GROUP_LOTFRONTAGE = [
            "Street",
            "LotShape",
            "LandContour",
            "Neighborhood",
        ]

    def fit(self, X, y=None):
        self.mappingLotFrontage = X.groupby(self.FEATURES_GROUP_LOTFRONTAGE)["LotFrontage"].mean()
        self.globalMeanLotFrontage = X["LotFrontage"].mean()
        #Calculate mode for BsmtExposure and Electrical
        if 'Electrical' in X.columns and not X['Electrical'].dropna().empty:
            self.modeElectrical = X['Electrical'].mode()[0]
        if 'BsmtExposure' in X.columns and not X['BsmtExposure'].dropna().empty:
            self.modeBsmtExposure = X['BsmtExposure'].mode()[0]
        return self

    def transform(self, X):
        xDF = X.copy()

        # Imputer PoolQC
        xDF.loc[xDF["PoolArea"] == 0, "PoolQC"] = "NA"
        # Imputer MiscFeature
        xDF.loc[xDF["MiscVal"] == 0, "MiscFeature"] = "NA"
        # Imputer MasVnrType
        xDF.loc[xDF["MasVnrArea"] == 0, "MasVnrType"] = "None"
        # Imputer FireplaceQu
        xDF.loc[xDF["Fireplaces"] == 0, "FireplaceQu"] = "NA"

        # Imputer Garage
        listMissingGarageCategorical = [
            "GarageFinish",
            "GarageQual",
            "GarageType",
            "GarageCond",
        ]
        for col in listMissingGarageCategorical:
            if col in xDF.columns:
                xDF.loc[xDF["GarageArea"] == 0, col] = "NA"
        xDF.loc[xDF["GarageArea"] == 0, "GarageYrBlt"] = 0

        # Imputer basement
        listMissingBasementCategorical = [
            "BsmtFinType1",
            "BsmtFinType2",
            "BsmtExposure",
            "BsmtCond",
            "BsmtQual",
        ]
        for col in listMissingBasementCategorical:
            if col in xDF.columns:
                xDF.loc[xDF["TotalBsmtSF"] == 0, col] = "NA"
        xDF.loc[(xDF['BsmtFinSF1'] != 0) & (xDF['BsmtUnfSF'] != 0) & (xDF['TotalBsmtSF'] != 0), 'BsmtFinType1'] = 'Unf'
        xDF.loc[(xDF['BsmtFinSF2'] != 0) & (xDF['BsmtUnfSF'] != 0) & (xDF['TotalBsmtSF'] != 0), 'BsmtFinType2'] = 'Unf'

        # Imputer LotFrontages
        lf_mapped_values = xDF.set_index(self.FEATURES_GROUP_LOTFRONTAGE).index.map(
            self.mappingLotFrontage
        )
        lf_mapped_series = pd.Series(lf_mapped_values, index=xDF.index)
        xDF["LotFrontage"] = xDF["LotFrontage"].fillna(lf_mapped_series)
        xDF["LotFrontage"] = xDF["LotFrontage"].fillna(self.globalMeanLotFrontage)

        xDF['Electrical'] = xDF['Electrical'].fillna(self.modeElectrical)
        xDF['BsmtExposure'] = xDF['BsmtExposure'].fillna(self.modeBsmtExposure)

        return xDF

cat_simpleImputer = ["Alley", "Fence"]
num_simpleImputer = ["MasVnrArea"]

pipe_preprocessing = Pipeline(
    [
        ("imputer_cat", SimpleImputer(strategy="constant", fill_value="NA", missing_values=None)),
        ("imputer_num", SimpleImputer(strategy="constant", fill_value=0, missing_values=None)),
        ("data_imputer", DataImputer()),
    ]
)
featuresOneHot = ["MiscFeature", "Alley"]
featuresOrdinal = ["PoolQC", "Fence"]
mappingPoolQC = ['NA', 'Fa', 'TA', 'Gd', 'Ex']
mappingFence = ['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']

""" pipe_featureCodification = pipeline([
    ("ordinalPoolQC", OrdinalEncoder(categories=[mappingOrdinal], handle_unknown='use_encoded_value', unknown_value=-1)),
]) """