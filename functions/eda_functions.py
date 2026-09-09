import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import (
    mannwhitneyu,
    spearmanr,
    ttest_ind,
    ks_2samp,
    pointbiserialr,
    pearsonr,
    permutation_test,
    chi2_contingency,
    power_divergence,
    fisher_exact,
    shapiro,
    kstest,
    skew,
    kurtosis,
    chi2,
    normaltest,
    anderson,
)
from scipy.stats.contingency import association


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


# --------------------------------------------------------------------
# Mann-Whitney p-value: 0.4521  → ¿Hay diferencia entre grupos?
# Spearman p-value:    0.9823  → ¿Hay relación monótona?
# Pearson p-value:     0.8345  → ¿Hay relación lineal?
# --------------------------------------------------------------------
# Mann-Whitney U Test:
# Comparación de distribuciones entre grupos con datos no paramétricos
# 1. Combina todos los datos
# 2. Asigna rangos a los datos combinados
# 3. Divide los datos en dos grupos: presentes y ausentes (por rangos)
# 4. Calcula la estadística U de Mann-Whitney para cada grupo y escoge el menor
# 5. Calcula el valor P
# --------------------------------------------------------------------
# Kolmogorov-Smirnov: Comparación de distribuciones entre dos grupos, sensible a diferencias en forma y dispersión
# --------------------------------------------------------------------
# T-student: Comparación de medias entre dos grupos, asume normalidad y varianzas iguales
# --------------------------------------------------------------------
# Permutation Test: Comparación de medias entre dos grupos mediante reordenamiento aleatorio de los datos, no asume normalidad
# --------------------------------------------------------------------
# COHEN'S: Magnitud de la diferencia entre grupos
# d < 0.2   | Muy pequeña | La diferencia entre grupos es mínima
# 0.2 - 0.5 | Pequeña    | La diferencia entre grupos es pequeña
# 0.5 - 0.8 | Moderada   | La diferencia entre grupos es moderada
# d >= 0.8  | Grande     | La diferencia entre grupos es grande
# --------------------------------------------------------------------
# SPEARMANR: Dirección de la correlación
# 0.0 - 0.1 | Insignificante | Prácticamente no hay relación
# 0.1 - 0.3 | Débil          | Hay una ligera tendencia
# 0.3 - 0.6 | Moderada       | La relación es claramente perceptible
# 0.6 - 0.8 | Fuerte         | La relación es muy clara
# 0.8 - 1.0 | Muy fuerte     | Casi una relación perfecta
# rho= +1, correlacion perfecta positiva; a mayor valor de la feature, mas missing
# rho= -1, correlacion perfecta negativa; a mayor valor de la feature, menos missing
# rho= 0, sin correlacion;
# --------------------------------------------------------------------
# PEARSON: Dirección y fuerza de la correlación lineal
# 0.0 - 0.1 | Insignificante | Prácticamente no hay relación
# 0.1 - 0.3 | Débil          | Hay una ligera tendencia
# 0.3 - 0.6 | Moderada       | La relación es claramente perceptible
# 0.6 - 0.8 | Fuerte         | La relación es muy clara
# 0.8 - 1.0 | Muy fuerte     | Casi una relación perfecta
# r= +1, correlacion perfecta positiva; a mayor valor de la feature, mas missing
# r= -1, correlacion perfecta negativa; a mayor valor de la feature, menos missing
# r= 0, sin correlacion;
# --------------------------------------------------------------------
class ApplyNumericTest:
    def __init__(self, df, missingFeature="", alpha=0.05, onlyTrue=False):
        self.useData = df.copy()
        self.missingFeature = missingFeature
        self.missingFeature_M = f"{missingFeature}_M"
        self.alpha = alpha
        self.onlyTrue = onlyTrue
        self.cols = []

    def define_groups(self):
        self.useData[self.missingFeature_M] = self.useData[self.missingFeature].isna()
        self.cols = self.useData.select_dtypes(include=[np.number]).columns.tolist()
        if "Id" in self.cols:
            self.cols.remove("Id")

    def __split_groups(self, col):
        MIN_ABSOLUTE_GROUP_SIZE = 3
        missing = self.useData.loc[
            self.useData[self.missingFeature].isna(), col
        ].dropna()
        present = self.useData.loc[
            self.useData[self.missingFeature].notna(), col
        ].dropna()
        len_missing = len(missing)
        len_present = len(present)
        GREATER_THAN_THE_MINIMUM = (
            len_present >= MIN_ABSOLUTE_GROUP_SIZE
            and len_missing >= MIN_ABSOLUTE_GROUP_SIZE
        )
        return present, missing, len_present, len_missing, GREATER_THAN_THE_MINIMUM

    def __config_output(self, res):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        if self.onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df

    def mann_whitney_u(self):
        self.define_groups()
        res = []
        for col in self.cols:
            present, missing, len_present, len_missing, GREATER_THAN_THE_MINIMUM = (
                self.__split_groups(col)
            )
            if GREATER_THAN_THE_MINIMUM:
                stat, p_value = mannwhitneyu(present, missing, alternative="two-sided")
                # COHEN'S D
                meanPresent = present.mean()
                meanMissing = missing.mean()
                std1Present = present.std(ddof=1)
                std1Missing = missing.std(ddof=1)
                pooled_std = np.sqrt(
                    (
                        (len_present - 1) * std1Present**2
                        + (len_missing - 1) * std1Missing**2
                    )
                    / (len_present + len_missing - 2)
                )
                cohen_d = (
                    (meanPresent - meanMissing) / pooled_std if pooled_std > 0 else 0
                )
                # cohen_d: Magnitud de la diferencia entre grupos

                # SPEARMANR
                rho, p_value_rho = spearmanr(
                    self.useData[col],
                    self.useData[self.missingFeature_M],
                    nan_policy="omit",
                )

                res.append(
                    {
                        "name_feature": col,
                        "mann_whitney_u": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                        "cohen_d": cohen_d.round(
                            2
                        ),  # Valor de la magnitud de la diferencia entre grupos
                        "spearman_rho": rho.round(
                            2
                        ),  # Direccion de la relacion monotona
                        "spearman_p_value": p_value_rho.round(
                            2
                        ),  # Hay relacion Monotona?
                    }
                )
        return self.__config_output(res)
    
    def kolmogorov_smirnov(self):
        self.define_groups()
        res = []
        for col in self.cols:
            present, missing, len_present, len_missing, GREATER_THAN_THE_MINIMUM = (
                self.__split_groups(col)
            )
            

            if GREATER_THAN_THE_MINIMUM:
                stat, p_value = ks_2samp(present, missing)
                #COHEN'S D
                meanPresent = present.mean()
                meanMissing = missing.mean()
                std1Present = present.std(ddof=1)
                std1Missing = missing.std(ddof=1)
                pooled_std = np.sqrt(
                    (
                        (len_present - 1) * std1Present**2
                        + (len_missing - 1) * std1Missing**2
                    )
                    / (len_present + len_missing - 2)
                )
                cohen_d = (
                    (meanPresent - meanMissing) / pooled_std if pooled_std > 0 else 0
                )
                #SPEARMAN
                rho, p_value_rho = spearmanr(
                    self.useData[col],
                    self.useData[self.missingFeature_M],
                    nan_policy="omit",
                )
                res.append(
                    {
                        "name_feature": col,
                        "ks_stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                        "cohen_d": cohen_d.round(2),
                        "spearman_rho": rho.round(2),
                        "spearman_p_value": p_value_rho.round(2),
                    }
                )
                
        return self.__config_output(res)

    def t_student(self):
        self.define_groups()
        res = []
        for col in self.cols:
            present, missing, len_present, len_missing, GREATER_THAN_THE_MINIMUM = (
                self.__split_groups(col)
            )
            if GREATER_THAN_THE_MINIMUM:
                stat, p_value = ttest_ind(present, missing, equal_var=False)
                # COHEN'S D
                meanPresent = present.mean()
                meanMissing = missing.mean()
                std1Present = present.std(ddof=1)
                std1Missing = missing.std(ddof=1)
                pooled_std = np.sqrt(
                    (
                        (len_present - 1) * std1Present**2
                        + (len_missing - 1) * std1Missing**2
                    )
                    / (len_present + len_missing - 2)
                )
                cohen_d = (
                    (meanPresent - meanMissing) / pooled_std if pooled_std > 0 else 0
                )

                # PEARSON
                r, p_value_r = pearsonr(
                    self.useData[col],
                    self.useData[self.missingFeature_M],
                    nan_policy="omit",
                )

                res.append(
                    {
                        "name_feature": col,
                        "t_stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                        "cohen_d": cohen_d.round(2),
                        "pearson_r": r.round(2),  # Direccion de la relacion lineal
                        "pearson_p_value": p_value_r.round(2),  # Hay relacion lineal?
                    }
                )
        return self.__config_output(res)

    def permutation_test(self):
        self.define_groups()
        res = []
        for col in self.cols:
            present, missing, len_present, len_missing, GREATER_THAN_THE_MINIMUM = (
                self.__split_groups(col)
            )
            if GREATER_THAN_THE_MINIMUM:
                stat, p_value = permutation_test(
                    present,
                    missing,
                    num_rounds=10000,
                    alternative="two-sided",
                    random_state=42,
                )

                # COHEN'S D
                meanPresent = present.mean()
                meanMissing = missing.mean()
                std1Present = present.std(ddof=1)
                std1Missing = missing.std(ddof=1)
                pooled_std = np.sqrt(
                    (
                        (len_present - 1) * std1Present**2
                        + (len_missing - 1) * std1Missing**2
                    )
                    / (len_present + len_missing - 2)
                )
                cohen_d = (
                    (meanPresent - meanMissing) / pooled_std if pooled_std > 0 else 0
                )

                # SPEARMAN'S RHO
                rho, p_value_rho = spearmanr(
                    self.useData[col],
                    self.useData[self.missingFeature_M],
                    nan_policy="omit",
                )

                res.append(
                    {
                        "name_feature": col,
                        "permutation_stat": stat,
                        "p_value": p_value,
                        "evidence_MAR": p_value < self.alpha,
                        "cohen_d": cohen_d.round(2),
                        "spearman_rho": rho.round(
                            2
                        ),  # Direccion de la relacion monotona
                        "spearman_p_value": p_value_rho.round(
                            2
                        ),  # Hay relacion monotona?
                    }
                )
        return self.__config_output(res)


# --------------------------------------------------------------------
# Chi-Squared Test: Evaluación de independencia entre variables categóricas
# 1. Construye una tabla de contingencia con las frecuencias observadas
# 2. Calcula las frecuencias esperadas bajo la hipótesis de independencia
# 3. Calcula la estadística Chi-cuadrado y el valor P
# 4. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# --------------------------------------------------------------------
# G Test: Evaluación de independencia entre variables categóricas
# 1. Construye una tabla de contingencia con las frecuencias observadas
# 2. Calcula las frecuencias esperadas bajo la hipótesis de independencia
# 3. Calcula la estadística G y el valor P
# 4. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# --------------------------------------------------------------------
# Fisher's Exact Test: Evaluación de independencia entre variables categóricas (2x2)
# 1. Construye una tabla de contingencia 2x2 con las frecuencias observadas
# 2. Calcula la probabilidad exacta de obtener la tabla observada
#    bajo la hipótesis de independencia
# 3. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# --------------------------------------------------------------------
# CRAMER'S V: Magnitud de la asociación entre variables categóricas
# 0.0 - 0.1 | Insignificante | Prácticamente no hay relación
# 0.1 - 0.3 | Débil          | Hay una ligera asociación
# 0.3 - 0.6 | Moderada       | La asociación es claramente perceptible
# 0.6 - 0.8 | Fuerte         | La asociación es muy clara
# 0.8 - 1.0 | Muy fuerte     | Casi una asociación perfecta
# V= +1, asociacion perfecta positiva; a mayor valor de la feature, mas missing
# V= -1, asociacion perfecta negativa; a mayor valor de la feature, menos missing
# V= 0, sin asociacion;
# --------------------------------------------------------------------
# THEIL'S U: Magnitud de la asociación entre variables categóricas
# U = 0, sin asociacion; U = 1, asociacion perfecta
# < 0.1 | Insignificante | Prácticamente no hay relación
# 0.1 - 0.3 | Débil          | Hay una ligera asociación
# 0.3 - 0.6 | Moderada       | La asociación es claramente
# 0.6 - 0.8 | Fuerte         | La asociación es muy clara
# 0.8 - 1.0 | Muy fuerte     | Casi una asociación
# --------------------------------------------------------------------
# ODDS RATIO: Magnitud de la asociación entre variables categóricas (2x2)
# OR = 1, sin asociacion; OR > 1, asociacion positiva; OR < 1, asociacion negativa
# 0.0 - 1.0 | Negativa       | A mayor valor de la feature, menos missing
# 1.0 - 1.5 | Débil          | Hay una ligera asociación
# 1.5 - 3.0 | Moderada       | La asociación es claramente perceptible
# 3.0 - 5.0 | Fuerte         | La asociación es muy clara
# 5.0 - 10.0 | Muy fuerte     | Casi una asociación perfecta
# --------------------------------------------------------------------
class ApplyCatetogicalTest:
    def __init__(self, useData, missingFeature, alpha=0.05, onlyTrue=True):
        self.useData = useData
        self.missingFeature = missingFeature
        self.missingFeature_M = f"{missingFeature}_M"
        self.alpha = alpha
        self.onlyTrue = onlyTrue
        self.cols = []

    def define_cols(self):
        self.useData[self.missingFeature_M] = self.useData[self.missingFeature].isna()
        self.cols = self.useData.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if "Id" in self.cols:
            self.cols.remove("Id")

    def __create_contingency(self, col):
        contingency = pd.crosstab(
            self.useData[col], self.useData[self.missingFeature_M]
        )
        contingency.columns = ["present", "missing"]
        NO_SHORTAGE_VARIANCE = contingency.shape[1] < 2
        return contingency, NO_SHORTAGE_VARIANCE

    def __config_output(self, res):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        if self.onlyTrue:
            res_df = res_df[res_df["evidence_MAR"]]
        return res_df

    def chi2_cuadrado(self):
        res = []
        for col in self.cols:
            contingency, NO_SHORTAGE_VARIANCE = self.__create_contingency(col)
            if NO_SHORTAGE_VARIANCE:
                continue
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency.values)

            # CRAMER'S
            cramer_v = association(contingency.values, method="cramer")

            # Residuos estandarizados
            n = contingency.sum().sum()
            row_props = contingency.sum(axis=1) / n
            col_props = contingency.sum(axis=0) / n
            # Residuos estandarizados
            residuos_str = (contingency.values - expected) / np.sqrt(
                expected
                * (1 - row_props.values[:, None])
                * (1 - col_props.values[None, :])
            )
            # Extraer residuos de la columna "Faltante" (columna 1)
            residuos_faltante = residuos_str[:, 1]
            # Proporcion missing por categorias
            prop_by_cat = contingency["missing"] / contingency.sum(axis=1)

            res.append(
                {
                    "name_feature": col,
                    "chi2_stat": chi2_stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                    "cramer_v": cramer_v.round(2),  # Magnitud de la asociacion
                    "residuos_faltante": residuos_faltante.tolist(),
                    "categorias": contingency.index.tolist(),
                }
            )
        return self.__config_output(res)

    def fisher_exact(self):
        res = []
        for col in self.cols:
            contingency, NO_SHORTAGE_VARIANCE = self.__create_contingency(col)
            # check variance or if contingency is not 2x2
            if NO_SHORTAGE_VARIANCE or contingency.shape != (2, 2):
                continue
            # Fisher's Exact Test
            oddsratio, p_value = fisher_exact(contingency.values)

            # IC 95% para el odds ratio
            a, b = contingency.values[0, 0], contingency.values[0, 1]
            c, d = contingency.values[1, 0], contingency.values[1, 1]
            if b == 0 or c == 0:
                # Si b o c son cero, el odds ratio es infinito o cero, y no podemos calcular un IC
                a += 0.5
                b += 0.5
                c += 0.5
                d += 0.5
            log_or = np.log(oddsratio)
            se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            z = 1.96  # para un IC del 95%
            ic_lower = np.exp(log_or - z * se_log_or)
            ic_upper = np.exp(log_or + z * se_log_or)
            ic_incluye_1 = ic_lower <= 1 <= ic_upper

        res.append(
            {
                "name_feature": col,
                "odds_ratio": oddsratio,
                "p_value": p_value,
                "evidence_MAR": p_value < self.alpha,
                "ic_95_lower": round(ic_lower, 3),
                "ic_95_upper": round(ic_upper, 3),
                "ic_incluye_1": ic_incluye_1,
            }
        )
        return self.__config_output(res)

    def g_test(self):
        res = []
        for col in self.cols:
            contingency, NO_SHORTAGE_VARIANCE = self.__create_contingency(col)
            if NO_SHORTAGE_VARIANCE:
                continue
            # G-Test
            g_stat, p_value = power_divergence(
                contingency.values, lambda_="log-likelihood"
            )

            # Residuos estandarizados
            chi2_stat, p_value_chi2, dof, expected = chi2_contingency(
                contingency.values
            )
            n = contingency.sum().sum()
            row_props = contingency.sum(axis=1) / n
            col_props = contingency.sum(axis=0) / n

            residuos_str = (contingency.values - expected) / np.sqrt(
                expected
                * (1 - row_props.values[:, None])
                * (1 - col_props.values[None, :])
            )

            # THEIL'S U
            theils_u_value = self.__theils_u(contingency)

            # heatmap
            self.__create_headmap(contingency, col, residuos_str, p_value_chi2)

            res.append(
                {
                    "name_feature": col,
                    "g_stat": g_stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < self.alpha,
                    "theils_u": theils_u_value.round(2),  # Magnitud de la asociacion
                }
            )
        return self.__config_output(res)

    def __theils_u(self, contingency):
        table = contingency.values
        n = table.sum()

        # Probabilidades de conjuntas y marginales
        p_xy = table / n
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)

        # entropias
        # H(x)
        h_x = -np.sum(p_x * np.log2(p_x + 1e-10))
        # H(y)
        h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
        # H(x|y)
        h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
        # informacion mutua
        mi = h_x + h_y - h_xy

        if h_x == 0 and h_y == 0:
            return 0.0
        else:
            return 2 * mi / (h_x + h_y)

    def __create_headmap(self, contingency, col, residuos, p_value):
        # residuos estandarizados
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            residuos,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Residuos Estandarizados"},
            xticklabels=contingency.columns,
            yticklabels=contingency.index,
            vmin=-3,  # Límites para mejor visualización
            vmax=3,
        )
        plt.title(
            f"Residuos Estandarizados: {self.missingFeature} vs {col}\n"
            f"(p-value = {p_value:.4f})"
        )
        plt.xlabel(f"Feature with missing values: {self.missingFeature}")
        plt.ylabel(f"{col}")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# H0: La muestra proviene de una distribución normal
# ---------------------------------------------------------------------
# SHAPIRO-WILK TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística W de Shapiro-Wilk y el valor P
# 2. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n < 5000)
# Sensible a outliers y a la asimetría de la distribución
# ---------------------------------------------------------------------
# KOLMOGOROV-SMIRNOV TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística D de Kolmogorov-Smirnov y el valor P
# 2. Evalúa la significancia estadística comparando el valor P
#    con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n <= 5000) y distribuciones continuas
# ---------------------------------------------------------------------
# ANDERSON-DARLING TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística A de Anderson-Darling y los valores críticos
# 2. Evalúa la significancia estadística comparando la estadística A con los valores críticos
# 3. Si la estadística A es mayor que el valor crítico correspondiente al nivel de significancia (alpha), se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n >= 5000)
# ---------------------------------------------------------------------
# DAGOSTINO-PEARSON TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística D de D'Agostino-Pearson y el valor P
# 2. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# usar:
# muestras (n >= 20)
# ---------------------------------------------------------------------
# JARQUE-BERA TEST: Evaluación de normalidad de una distribución
# 1. Calcula la estadística JB de Jarque-Bera y el valor P
# 2. Evalúa la significancia estadística comparando el valor P con el nivel de significancia (alpha)
# 3. Si el valor P es menor que alpha, se rechaza la hipótesis nula de normalidad, indicando que la distribución no es normal.
# ---------------------------------------------------------------------
class NormalDistributionTest:
    def __init__(self, df, alpha=0.05):
        self.useData = df.copy()
        self.alpha = alpha
        self.cols = []
        self.define_cols()

    def define_cols(self, custom_cols = []):
        self.cols = self.useData.select_dtypes(include=[np.number]).columns.tolist()
        if "Id" in self.cols:
            self.cols.remove("Id")

    def __config_output(self, res):
        res_df = pd.DataFrame(res).sort_values(by="p_value")
        res_df["p_value"] = res_df["p_value"].round(5)
        return res_df
    
    def checkCustomCols(self, custom_cols=[]):
        if len(custom_cols) == 0:
            return self.cols
        else:
            result = all(e in self.cols for e in custom_cols)
            if result:
                return custom_cols
        return self.cols

    def shapiro_wilk_test(self, custom_cols = []):
        res = []
        list_cols = self.checkCustomCols(custom_cols)
        for col in list_cols:
            stat, p_value = shapiro(self.useData[col].dropna())
            res.append(
                {
                    "name_feature": col,
                    "shapiro_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self.__config_output(res)

    def kolmogorov_smirnov(self, custom_cols = []):
        res = []
        list_cols = self.checkCustomCols(custom_cols)
        for col in list_cols:
            mu, sigma = self.useData[col].mean(), self.useData[col].std()
            stat, p_value = kstest(self.useData[col], dist="norm", args=(mu, sigma))
            res.append(
                {
                    "name_feature": col,
                    "ks_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self.__config_output(res)

    def anderson_darling(self, custom_cols = []):
        res = []
        list_cols = self.checkCustomCols(custom_cols)
        for col in list_cols:
            result = anderson(self.useData[col].dropna(), dist="norm")
            stat = result.statistic
            critical_values = result.critical_values
            significance_level = result.significance_level

            idx_05 = list(significance_level).index(
                5.0
            )  # Nivel de significancia del 5%
            normal = stat < critical_values[idx_05]
            evidence_non_normality = (
                stat > critical_values[idx_05]
            )  # Usando el nivel de significancia del 5%

            res.append(
                {
                    "name_feature": col,
                    "ad_stat": stat,
                    "critical_value_95": critical_values[idx_05],
                    "normal": normal,
                    "interpretacion": "No normal" if not normal else "Normal",
                    "critical_values": dict(zip(significance_level, critical_values)),
                    "evidence_non_normality": evidence_non_normality,
                }
            )
        return self.__config_output(res)

    def dagostino_pearson(self, custom_cols = []):
        res = []
        list_cols = self.checkCustomCols(custom_cols)
        for col in list_cols:
            stat, p_value = normaltest(self.useData[col].dropna())
            res.append(
                {
                    "name_feature": col,
                    "dagostino_stat": stat,
                    "p_value": p_value,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self.__config_output(res)

    def jarque_bera(self, custom_cols = []):
        res = []
        list_cols = self.checkCustomCols(custom_cols)
        for col in list_cols:
            n = len(self.useData[col].dropna())
            skewness = skew(self.useData[col].dropna())
            kurtosiss = kurtosis(self.useData[col].dropna(), fisher=True)

            jb_stat = (n / 6) * (skewness**2 + (kurtosiss**2) / 4)
            p_value = 1 - chi2.cdf(jb_stat, df=2)
            normal = p_value >= self.alpha
            res.append(
                {
                    "name_feature": col,
                    "jb_stat": jb_stat,
                    "p_value": p_value,
                    "skewness": skewness,
                    "kurtosis": kurtosiss,
                    "normal": normal,
                    "evidence_non_normality": p_value < self.alpha,
                    "interpretacion": "No normal" if p_value < self.alpha else "Normal",
                }
            )
        return self.__config_output(res)
