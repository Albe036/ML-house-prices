import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind, ks_2samp, pointbiserialr

useData = pd.read_csv(
    "C:\\Users\\albeiro\\Documents\\GitHub\\ML-house-prices\\data\\raw\\train.csv"
)


def list_missing_values(df):
    df_train = df.copy()
    missing_data = df_train.isnull().sum()
    missing_data = pd.DataFrame(
        missing_data[missing_data > 0], columns=["missing counts"]
    )
    missing_data["percentage(%)"] = np.round(
        missing_data["missing counts"] / df_train.shape[0] * 100, 2
    )
    missing_data = missing_data.sort_values(by="percentage(%)", ascending=False)
    display(missing_data)


# MCAR: missing completely at random (completamente al azar)
# MNAR: missing not at random (no al azar)
# MAR: missing at random (al azar)
def test_mcar(df, variable_con_nulos, target_var="SalePrice"):
    # 1. Crear una copia temporal con la bandera de nulos
    temp_df = df.copy()
    temp_df["is_null"] = temp_df[variable_con_nulos].isnull()

    # 2. Separar los dos grupos
    grupo_nulos = temp_df[temp_df["is_null"] == True][target_var]
    grupo_datos = temp_df[temp_df["is_null"] == False][target_var]

    # 3. Visualización
    plt.figure(figsize=(12, 5))

    # Gráfico de densidad (KDE)
    plt.subplot(1, 2, 1)
    sns.kdeplot(grupo_datos, label="Con Datos", fill=True)
    sns.kdeplot(grupo_nulos, label="Nulos (NaN)", fill=True)
    plt.title(f"Distribución de {target_var}\nsegún nulidad de {variable_con_nulos}")
    plt.legend()

    # Boxplot para ver medianas y outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(data=temp_df, x="is_null", y=target_var)
    plt.title(f"Comparación de Medianas")

    plt.tight_layout()
    plt.show()

    # 4. Prueba Estadística (Mann-Whitney U)
    # Es mejor que la t-test porque no asume que los precios son normales
    stat, p_value = mannwhitneyu(grupo_datos, grupo_nulos)

    print(f"--- Diagnóstico para {variable_con_nulos} ---")
    print(f"P-Valor de la prueba Mann-Whitney: {p_value:.4f}")

    if p_value > 0.05:
        print("Resultado: No hay diferencia significativa. Posible MCAR (Aleatorio).")
    else:
        print(
            "Resultado: Diferencia significativa detectada. Es MAR o MNAR (No aleatorio)."
        )


class HypothesisTestNumeric:
    def __init__(self, df, baseFeature, onlyTrue=False, alpha=0.05):
        self.useData = df.copy()
        self.baseFeature = baseFeature
        self.baseFeature_M = f"{baseFeature}_M"
        self.numeric_cols = []
        self.onlyTrue = onlyTrue
        self.alpha = alpha
        self.define_groups()

    def define_groups(self):
        self.useData[self.baseFeature_M] = self.useData[self.baseFeature].isna()
        self.numeric_cols = self.useData.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if "Id" in self.numeric_cols:
            self.numeric_cols.remove("Id")

    def __split_groups(self, col):
        dataMissing = self.useData.loc[
            self.useData[self.baseFeature].isna(), col
        ].dropna()
        dataNotMissing = self.useData.loc[
            self.useData[self.baseFeature].notna(), col
        ].dropna()
        return dataMissing, dataNotMissing

    def __config_output(self, res):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        if self.onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df

    def mann_whitney_u(self):
        res = []
        for col in self.numeric_cols:
            m_1, m_0 = self.__split_groups(col)
            if len(m_1) > 0 and len(m_0) > 0:
                stat, p_value = mannwhitneyu(m_1, m_0, alternative="two-sided")
                res.append(
                    {
                        "name_feature": col,
                        "mann_whitney_U": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                    }
                )
        return self.__config_output(res)

    def t_student(self):
        res = []
        for col in self.numeric_cols:
            m_1, m_0 = self.__split_groups(col)
            if len(m_1) > 0 and len(m_0) > 0:
                stat, p_value = ttest_ind(
                    m_1, m_0, equal_var=False, alternative="two-sided"
                )
                res.append(
                    {
                        "name_feature": col,
                        "t_student": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                    }
                )
        return self.__config_output(res)

    def kolmogorov_smirnov(self):
        res = []
        for col in self.numeric_cols:
            m_1, m_0 = self.__split_groups(col)
            if len(m_1) > 0 and len(m_0) > 0:
                stat, p_value = ks_2samp(m_1, m_0)
                res.append(
                    {
                        "name_feature": col,
                        "kolmogorov_smirnov": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                    }
                )
        return self.__config_output(res)

    def point_biserial(self):
        res = []
        for col in self.numeric_cols:
            m_1, m_0 = self.__split_groups(col)
            if len(m_1) > 0 and len(m_0) > 0:
                stat, p_value = pointbiserialr(
                    self.useData[self.baseFeature_M], self.useData[col]
                )
                res.append(
                    {
                        "name_feature": col,
                        "point_biserial": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                    }
                )
        return self.__config_output(res)

    def spearman(self):
        res = []
        for col in self.numeric_cols:
            m_1, m_0 = self.__split_groups(col)
            if len(m_1) > 0 and len(m_0) > 0:
                stat, p_value = spearmanr(
                    self.useData[self.baseFeature_M], self.useData[col]
                )
                res.append(
                    {
                        "name_feature": col,
                        "spearman": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                    }
                )
        return self.__config_output(res)

class HypothesisTestCategorical:

    def __init__(self, df, baseFeature, onlyTrue=False, alpha=0.05):
        self.useData = df.copy()
        self.baseFeature = baseFeature
        self.baseFeature_M = f"{baseFeature}_M"
        self.onlyTrue = onlyTrue
        self.alpha = alpha
        self.categorical_cols = []

    def define_groups(self):
        self.useData[self.baseFeature_M] = self.useData[self.baseFeature].isna()
        self.numeric_cols = self.useData.select_dtypes(
            include=[np.]
        ).columns.tolist()
        if "Id" in self.numeric_cols:
            self.numeric_cols.remove("Id")