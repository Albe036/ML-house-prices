def impute_bsmt_features(df):
    bsmt_features = ['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    condition_not_basement = (df['TotalBsmtSF'] == 0) & (df['BsmtUnfSF'] == 0)
    df.loc[condition_not_basement, bsmt_features] = df.loc[condition_not_basement, bsmt_features].fillna('NA')
    return df

def impute_basement_exposure(df):
    df['BsmtExposure'] = df['BsmtExposure'].fillna(df.groupby(['BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtFullBath', 'BsmtHalfBath'])['BsmtExposure']
                                                    .transform(lambda x: x.mode()[0] if not x.mode().empty else 'No'))
    return df

def impute_basement_type2(df):
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df.groupby(['BsmtCond', 'BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtExposure'])['BsmtFinType2']
                                                    .transform(lambda x: x.mode()[0] if not x.mode().empty else 'Unf'))
    return df