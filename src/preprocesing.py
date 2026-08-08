import pandas as pd
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

#imputacion de datos faltantes
def imputerPoolQC(values):
    x = pd.DataFrame(values, columns=['PoolQC', 'PoolArea'])
    x.loc[x['PoolArea'] == 0, 'PoolQC'] = 'NA'
    return x['PoolQC'].values

def imputerMiscFeature(values):
    x = pd.DataFrame(values, columns=['MiscFeature', 'MiscVal'])
    x.loc[x['MiscVal'] == 0, 'MiscFeature'] = 'NA'
    return x['MiscFeature'].values

def imputerAlley(values):
    x = pd.DataFrame(values, columns=['Alley'])
    x.loc[:, 'Alley'] = x['Alley'].fillna('NA')
    return x['Alley'].values

def imputerFence(values):
    x = pd.DataFrame(values, columns=['Fence'])
    x.loc[:, 'Fence'] = x['Fence'].fillna('NA')
    return x['Fence'].values


f_imputerPoolQC = FunctionTransformer(imputerPoolQC, validate=False)
f_imputerMiscFeature = FunctionTransformer(imputerMiscFeature, validate=False)
f_imputerAlley = FunctionTransformer(imputerAlley, validate=False)
f_imputerFence = FunctionTransformer(imputerFence, validate=False)

pipeline_customTransformer = Pipeline(
    steps=[
        ('imputerPoolQC', f_imputerPoolQC),
        ('imputerMiscFeature', f_imputerMiscFeature),
        ('imputerAlley', f_imputerAlley),
        ('imputerFence', f_imputerFence)
    ]
)