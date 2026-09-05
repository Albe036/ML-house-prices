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
        dataMissing = self.useData.loc[self.useData[self.baseFeature].isna(), col].dropna()
        dataNotMissing = self.useData.loc[self.useData[self.baseFeature].notna(), col].dropna()
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

#--------------------------------------------------------------------
# Mann-Whitney U Test: 
# Comparación de distribuciones entre grupos con datos no paramétricos
# 1. Combina todos los datos
# 2. Asigna rangos a los datos combinados
# 3. Divide los datos en dos grupos: presentes y ausentes (por rangos)
# 4. Calcula la estadística U de Mann-Whitney para cada grupo y escoge el menor
# 5. Calcula el valor P
#--------------------------------------------------------------------
# COHEN'S: Magnitud de la diferencia entre grupos                    
# d < 0.2   | Muy pequeña | La diferencia entre grupos es mínima     
# 0.2 - 0.5 | Pequeña    | La diferencia entre grupos es pequeña     
# 0.5 - 0.8 | Moderada   | La diferencia entre grupos es moderada    
# d >= 0.8  | Grande     | La diferencia entre grupos es grande      
#--------------------------------------------------------------------
# SPEARMANR: Dirección de la correlación                             
# 0.0 - 0.1 | Insignificante | Prácticamente no hay relación         
# 0.1 - 0.3 | Débil          | Hay una ligera tendencia              
# 0.3 - 0.6 | Moderada       | La relación es claramente perceptible 
# 0.6 - 0.8 | Fuerte         | La relación es muy clara              
# 0.8 - 1.0 | Muy fuerte     | Casi una relación perfecta    
# rho= +1, correlacion perfecta positiva; a mayor valor de la feature, mas missing
# rho= -1, correlacion perfecta negativa; a mayor valor de la feature, menos missing
# rho= 0, sin correlacion;        
#--------------------------------------------------------------------
class ApplyNumericTest:
    def __init__(self, df, missingFeature="", alpha=0.05, onlyTrue=False):
        self.useData = df.copy()
        self.missingFeature = missingFeature
        self.missingFeature_M = f"{missingFeature}_M"
        self.alpha = alpha
        self.onlyTrue = onlyTrue
        self.numeric_cols = []

    def define_groups(self):
        self.useData[self.missingFeature_M] = self.useData[self.missingFeature].isna()
        self.numeric_cols = self.useData.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if "Id" in self.numeric_cols:
            self.numeric_cols.remove("Id")

    def __split_groups(self, col):
        missing = self.useData.loc[self.useData[self.missingFeature].isna(), col].dropna()
        present = self.useData.loc[self.useData[self.missingFeature].notna(), col].dropna()
        return present, missing

    def __config_output(self, res):
            res_df = pd.DataFrame(res).sort_values(by="p_value")
            res_df["p_value"] = res_df["p_value"].round(5)
            if self.onlyTrue:
                res_df = res_df[res_df["evidence_MAR"]]
            return res_df

    def mann_whitney_u(self):
        MIN_ABSOLUTE_GROUP_SIZE = 3
        self.define_groups()
        res = []
        for col in self.numeric_cols:
            present, missing = self.__split_groups(col)
            len_present = len(present)
            len_missing = len(missing)
            if len_present >= MIN_ABSOLUTE_GROUP_SIZE and len_missing >= MIN_ABSOLUTE_GROUP_SIZE:
                stat, p_value = mannwhitneyu(present, missing, alternative='two-sided')
                #COHEN'S D
                meanPresent = present.mean()
                meanMissing = missing.mean()
                std1Present = present.std(ddof=1)
                std1Missing = missing.std(ddof=1)
                pooled_std = np.sqrt(((len_present - 1) * std1Present ** 2 + (len_missing - 1) * std1Missing ** 2) / (len_present + len_missing - 2))
                cohen_d = (meanPresent - meanMissing) / pooled_std if pooled_std > 0 else 0
                #cohen_d: Magnitud de la diferencia entre grupos

                #SPEARMANR
                rho, p_value_rho = spearmanr(self.useData[col], self.useData[self.missingFeature_M], nan_policy='omit')

                res.append(
                    {
                        "name_feature": col,
                        "mann_whitney_u": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                        "cohen_d": cohen_d.round(2),
                        "spearman_rho": rho.round(2)
                    }
                )
        return self.__config_output(res)

    def t_student(self):
        MIN_ABSOLUTE_GROUP_SIZE = 3
        self.define_groups()
        res = []
        for col in self.numeric_cols:
            present, missing = self.__split_groups(col)
            len_present = len(present)
            len_missing = len(missing)
            if len_present > MIN_ABSOLUTE_GROUP_SIZE and len_missing > MIN_ABSOLUTE_GROUP_SIZE:
                stat, p_value = ttest_ind(present, missing, equal_var=False)
                #COHEN'S D
                meanPresent = present.mean()
                meanMissing = missing.mean()
                std1Present = present.std(ddof=1)
                std1Missing = missing.std(ddof=1)
                pooled_std = np.sqrt(((len_present - 1) * std1Present ** 2 + (len_missing - 1) * std1Missing ** 2) / (len_present + len_missing - 2))
                cohen_d = (meanPresent - meanMissing) / pooled_std if pooled_std > 0 else 0

                #SPEARMANR
                rho, p_value_rho = spearmanr(self.useData[col], self.useData[self.missingFeature_M], nan_policy='omit')

                res.append(
                    {
                        "name_feature": col,
                        "t_stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                        "cohen_d": cohen_d.round(2),
                        "spearman_rho": rho.round(2)
                    }
                )
        return self.__config_output(res)

