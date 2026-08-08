import pandas as pd
from pathlib import Path
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = BASE_DIR / 'data' / 'raw' / 'train.csv'
useTrain = pd.read_csv(TRAIN_PATH)

X = useTrain.drop(columns=['SalePrice'])
y = useTrain['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#imputacion de datos faltantes
def imputerPoolQC(rows):
    x = pd.DataFrame(rows, columns=['PoolQC', 'PoolArea'])
    x.loc[x['PoolArea'] == 0, 'PoolQC'] = 'NA'
    return x['PoolQC'].values

def imputerMiscFeature(rows):
    x = pd.DataFrame(rows, columns=['MiscFeature', 'MiscVal'])
    x.loc[x['MiscVal'] == 0, 'MiscFeature'] = 'NA'
    return x['MiscFeature'].values

def imputerAlley(rows):
    x = pd.DataFrame(rows, columns=['Alley'])
    x.loc[:, 'Alley'] = x['Alley'].fillna('NA')
    return x['Alley'].values

def imputerFence(rows):
    x = pd.DataFrame(rows, columns=['Fence'])
    x.loc[:, 'Fence'] = x['Fence'].fillna('NA')
    return x['Fence'].values

def imputerMasVnrArea(rows):
    x = pd.DataFrame(rows, columns=['MasVnrType', 'MasVnrArea'])
    x.loc[:,'MasVnrArea'] = x.loc[x['MasVnrType'].isnull(), 'MasVnrArea'].fillna(0)
    return x['MasVnrArea'].values

def imputerMasVnrType(rows):
    x = pd.DataFrame(rows, columns=['MasVnrType', 'MasVnrArea'])
    x.loc[x['MasVnrArea'] == 0, 'MasVnrType'] = 'None'
    return x['MasVnrType'].values

transformer_imputer = Pipeline([
    ('imputerPoolQC', FunctionTransformer(imputerPoolQC, validate=False)),
    ('imputerMiscFeature', FunctionTransformer(imputerMiscFeature, validate=False)),
    ('imputerAlley', FunctionTransformer(imputerAlley, validate=False)),
    ('imputerFence', FunctionTransformer(imputerFence, validate=False)),
    ('imputerMasVnrArea', FunctionTransformer(imputerMasVnrArea, validate=False)),
    ('imputerMasVnrType', FunctionTransformer(imputerMasVnrType, validate=False))
])

res = transformer_imputer.named_steps['imputerPoolQC'].transform(X_train[['PoolQC', 'PoolArea']])

res = pd.DataFrame(res)
print(res)