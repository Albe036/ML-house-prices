import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from IPython.display import display


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


def mannwhitneyu_tstudent_method(df, m="", y="", onlyTrue=False, method=0):
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp, pointbiserialr

    MANN_WHITNEY_U = 0
    T_STUDENT = 1
    KOLMOGOROV_SMIRNOV = 2
    POINT_BISERIAL = 3
    SPEAR_MANR = 4

    useData = df.copy()
    # Creacion de variable indicadora
    nameBool = f"{m}_M"
    # Convertir la variable indicadora de ausencia a booleana
    useData[nameBool] = useData[m].isna()

    numeric_cols = useData.select_dtypes(include=[np.number]).columns.tolist()
    if "Id" in numeric_cols:
        numeric_cols.remove("Id")

    result = []

    for col in numeric_cols:
        groupWithOutMissing = useData.loc[useData[m].notna(), col].dropna()
        groupWithMissing = useData.loc[useData[m].isna(), col].dropna()

        if len(groupWithOutMissing) > 0 and len(groupWithMissing) > 0:
            stat_method = ""
            if method == MANN_WHITNEY_U:
                stat, p_value = mannwhitneyu(
                    groupWithMissing, groupWithOutMissing, alternative="two-sided"
                )
                stat_method = "Mann-Whitney U"
            elif method == T_STUDENT:
                stat, p_value = ttest_ind(
                    groupWithMissing,
                    groupWithOutMissing,
                    equal_var=False,
                    alternative="two-sided",
                )
                stat_method = "T-Student"
            elif method == KOLMOGOROV_SMIRNOV:
                stat, p_value = ks_2samp(
                    groupWithMissing, groupWithOutMissing, alternative="two-sided"
                )
                stat_method = "Kolmogorov-Smirnov"
            elif method == POINT_BISERIAL:
                stat, p_value = pointbiserialr(useData[nameBool], useData[col])
                stat_method = "Point-Biserial"
            elif method == SPEAR_MANR:
                from scipy.stats import spearmanr
                stat, p_value = spearmanr(useData[nameBool], useData[col])
                stat_method = "Spearman"
            result.append(
                {
                    "name_feature": col,
                    stat_method: stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < 0.05,
                }
            )
    df_res = pd.DataFrame(result).sort_values(by="p_value")
    df_res["p_value"] = df_res["p_value"].round(5)
    if onlyTrue:
        df_res = df_res[df_res["evidence_MAR"]]
    return df_res


class HypothesisTest:
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu, ttest_ind, ks_2samp, pointbiserialr
    def __init__(self, df, baseFeature, onlyTrue=False):
        self.useData = df.copy()
        self.baseFeature = baseFeature
        self.baseFeature_M = f"{baseFeature}_M"
        self.numeric_cols = None
        self.onlyTrue = onlyTrue

    def define_groups(self):
        self.useData[self.baseFeature_M] = self.useData[self.baseFeature].isna()
        self.numeric_cols = self.useData.select_dtypes(include=np.number).columns.tolist()
        if "Id" in self.numeric_cols:
            self.numeric_cols.remove("Id")

    def __split_groups(self, col):
        z = self.baseFeature_M
        dataMissing = self.useData.loc[self.useData[z].isna(), col]
        dataNotMissing = self.useData.loc[self.useData[z].notna(), col]
        return dataMissing, dataNotMissing

    def __config_output(self, res):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        if self.onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df

    def mann_whitney_u(self, dataMissing, dataNotMissing):
        z = self.baseFeature_M
        res = []
        for col in self.numeric_cols:
            m1, m0 = self.__split_groups(col)
            if len(m_1) > 0 and len(m_0) > 0:
                stat, p_value = mannwhitneyu(dataMissing, dataNotMissing, alternative="two-sided")
                result.append({
                    "name_feature": col,
                    "mann_whitney_U": stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < 0.05,
                })
        return self.__config_output(res)