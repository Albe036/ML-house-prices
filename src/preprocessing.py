import pandas as pd
from pathlib import Path
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
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
class CustomDataImputer(BaseEstimator, TransformerMixin):
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
        self.mappingLotFrontage = X.groupby(self.FEATURES_GROUP_LOTFRONTAGE)[
            "LotFrontage"
        ].mean()
        self.globalMeanLotFrontage = X["LotFrontage"].mean()
        # Calculate mode for BsmtExposure and Electrical
        if "Electrical" in X.columns and not X["Electrical"].dropna().empty:
            self.modeElectrical = X["Electrical"].mode()[0]
        if "BsmtExposure" in X.columns and not X["BsmtExposure"].dropna().empty:
            self.modeBsmtExposure = X["BsmtExposure"].mode()[0]
        return self

    def transform(self, X):
        xDF = X.copy()
        xDF["MSZoning"] = xDF["MSZoning"].replace("C (all)", "C")
        xDF["Exterior2nd"] = xDF["Exterior2nd"].replace("Wd Sdng", "WdSdng")
        xDF["Exterior2nd"] = xDF["Exterior2nd"].replace("Wd Shng", "WdShing")
        xDF["Exterior2nd"] = xDF["Exterior2nd"].replace("Brk Cmn", "BrkComm")
        xDF["Exterior2nd"] = xDF["Exterior2nd"].replace("CmentBd", "CemntBd")
        xDF["Exterior1st"] = xDF["Exterior1st"].replace("Wd Sdng", "WdSdng")
        xDF["Exterior1st"] = xDF["Exterior1st"].replace("Wd Shng", "WdShing")
        xDF["Exterior1st"] = xDF["Exterior1st"].replace("Brk Cmn", "BrkComm")
        xDF["Exterior1st"] = xDF["Exterior1st"].replace("CmentBd", "CemntBd")
        
        # Estandarizacion mayúsculas/minúsculas
        for col in xDF.columns:
            if xDF[col].dtype == "object":
                xDF[col] = xDF[col].str.lower()
        

        # Imputer PoolQC
        xDF.loc[xDF["PoolArea"] == 0, "PoolQC"] = "na"
        # Imputer MiscFeature
        xDF.loc[xDF["MiscVal"] == 0, "MiscFeature"] = "na"
        # Imputer MasVnrType
        xDF.loc[xDF["MasVnrArea"] == 0, "MasVnrType"] = "none"
        # Imputer FireplaceQu
        xDF.loc[xDF["Fireplaces"] == 0, "FireplaceQu"] = "na"

        # Imputer Garage
        listMissingGarageCategorical = [
            "GarageFinish",
            "GarageQual",
            "GarageType",
            "GarageCond",
        ]
        for col in listMissingGarageCategorical:
            if col in xDF.columns:
                xDF.loc[xDF["GarageArea"] == 0, col] = "na"
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
                xDF.loc[xDF["TotalBsmtSF"] == 0, col] = "na"
        xDF.loc[
            (xDF["BsmtFinSF1"] != 0)
            & (xDF["BsmtUnfSF"] != 0)
            & (xDF["TotalBsmtSF"] != 0),
            "BsmtFinType1",
        ] = "unf"
        xDF.loc[
            (xDF["BsmtFinSF2"] != 0)
            & (xDF["BsmtUnfSF"] != 0)
            & (xDF["TotalBsmtSF"] != 0),
            "BsmtFinType2",
        ] = "unf"

        # Imputer LotFrontages
        lf_mapped_values = xDF.set_index(self.FEATURES_GROUP_LOTFRONTAGE).index.map(
            self.mappingLotFrontage
        )
        lf_mapped_series = pd.Series(lf_mapped_values, index=xDF.index)
        xDF["LotFrontage"] = xDF["LotFrontage"].fillna(lf_mapped_series)
        xDF["LotFrontage"] = xDF["LotFrontage"].fillna(self.globalMeanLotFrontage)

        xDF["Electrical"] = xDF["Electrical"].fillna(self.modeElectrical)
        xDF["BsmtExposure"] = xDF["BsmtExposure"].fillna(self.modeBsmtExposure)

        return xDF


# Simple imputer
catSimpleImputer = ["Alley", "Fence"]
numSimpleImputer = ["MasVnrArea"]

simpleImputerTransformer = ColumnTransformer(
    transformers=[
        (
            "imputer_cat",
            SimpleImputer(strategy="constant", fill_value="NA"),
            catSimpleImputer,
        ),
        (
            "imputer_num",
            SimpleImputer(strategy="constant", fill_value=0),
            numSimpleImputer,
        ),
    ]
)

featuresOneHot = [
    "MiscFeature",
    "Alley",
    "GarageType",
    "Electrical",
    "BldgType",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "LandContour",
    "Street",
    "MSZoning",
    "HouseStyle",
    "Heating",
    "Exterior2nd",
    "RoofMatl",
    "RoofStyle",
    "Functional",
    "SaleType",
    "SaleCondition"
]
featMappingOrdinal = {
    "PoolQC": ["na", "fa", "ta", "gd", "ex"],
    "Fence": ["na", "mnww", "gdwo", "mnprv", "gdprv"],
    "MasVnrType": ["none", "brkcmn", "brkface", "stone"],
    "FireplaceQu": ["na", "po", "fa", "ta", "gd", "ex"],
    "GarageQual": ["na", "po", "fa", "ta", "gd", "ex"],
    "GarageFinish": ["na", "unf", "rfn", "fin"],
    "GarageCond": ["na", "po", "fa", "ta", "gd", "ex"],
    "BsmtExposure": ["na", "no", "mn", "av", "gd"],
    "BsmtFinType2": ["na", "unf", "lwq", "rec", "blq", "alq", "glq"],
    "BsmtQual": ["na", "po", "fa", "ta", "gd", "ex"],
    "BsmtFinType1": ["na", "unf", "lwq", "rec", "blq", "alq", "glq"],
    "BsmtCond": ["na", "po", "fa", "ta", "gd", "ex"],
    "LandSlope": ["gtl", "mod", "sev"],
    "LotShape": ["ir3", "ir2", "ir1", "reg"],
    "LotConfig": ["inside", "fr2", "corner", "fr3", "culdsac"],
    "Utilities": ["elo", "nosewa", "nosewr", "allpub"],
    "Foundation": ["slab", "brktil", "cblock", "stone", "wood", "pconc"],
    "ExterQual": ["po", "fa", "ta", "gd", "ex"],
    "ExterCond": ["po", "fa", "ta", "gd", "ex"],
    "CentralAir": ["n", "y"],
    "HeatingQC": ["po", "fa", "ta", "gd", "ex"],
    "KitchenQual": ["po", "fa", "ta", "gd", "ex"],
    "PavedDrive": ["n", "p", "y"],
}

cols_ordinal = list(featMappingOrdinal.keys())
cats_ordinal = list(featMappingOrdinal.values())
ordinalEnconder = Pipeline([
    ('ordinal', OrdinalEncoder(categories=cats_ordinal, dtype=float, handle_unknown='use_encoded_value', unknown_value=-1)),
    ("scaler", StandardScaler())
])
oneHotEncoder = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int))
])
featuresContinuas = [col for col in X.columns if col not in featuresOneHot and col not in cols_ordinal]

numericTransformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

processorEncoder = ColumnTransformer(
    transformers=[
        ("ordinal", ordinalEnconder, cols_ordinal),
        ("onehot", oneHotEncoder, featuresOneHot),
        ("continua", numericTransformer, featuresContinuas)
    ],
    remainder="drop",
)


AllPreprocessing = Pipeline(
    [
        ("data_imputer", CustomDataImputer()),
        ("simple_imputer", simpleImputerTransformer),
        ("encoder", processorEncoder),
    ]
)