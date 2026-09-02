import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from IPython.display import display

def list_missing_values(df):
    df_train = df.copy()
    missing_data = df_train.isnull().sum()
    missing_data = pd.DataFrame(missing_data[missing_data > 0], columns=['missing counts'])
    missing_data['percentage(%)'] = np.round(missing_data['missing counts'] / df_train.shape[0] * 100, 2)
    missing_data = missing_data.sort_values(by='percentage(%)', ascending=False)
    display(missing_data)
    

#MCAR: missing completely at random (completamente al azar)
#MNAR: missing not at random (no al azar)
#MAR: missing at random (al azar)
def test_mcar(df, variable_con_nulos, target_var='SalePrice'):
    # 1. Crear una copia temporal con la bandera de nulos
    temp_df = df.copy()
    temp_df['is_null'] = temp_df[variable_con_nulos].isnull()
    
    # 2. Separar los dos grupos
    grupo_nulos = temp_df[temp_df['is_null'] == True][target_var]
    grupo_datos = temp_df[temp_df['is_null'] == False][target_var]
    
    # 3. Visualización
    plt.figure(figsize=(12, 5))
    
    # Gráfico de densidad (KDE)
    plt.subplot(1, 2, 1)
    sns.kdeplot(grupo_datos, label='Con Datos', fill=True)
    sns.kdeplot(grupo_nulos, label='Nulos (NaN)', fill=True)
    plt.title(f'Distribución de {target_var}\nsegún nulidad de {variable_con_nulos}')
    plt.legend()
    
    # Boxplot para ver medianas y outliers
    plt.subplot(1, 2, 2)
    sns.boxplot(data=temp_df, x='is_null', y=target_var)
    plt.title(f'Comparación de Medianas')
    
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
        print("Resultado: Diferencia significativa detectada. Es MAR o MNAR (No aleatorio).")


def mannwhitneyu_method(df, m="", y="", onlyTrue=False):
    import pandas as pd
    import numpy as np
    from scipy.stats import mannwhitneyu

    useData = df.copy()
    #Creacion de variable indicadora
    nameBool = f"{m}_M"
    #Convertir la variable indicadora de ausencia a booleana
    useData[nameBool] = useData[m].isna()

    numeric_cols = useData.select_dtypes(include=[np.number]).columns.tolist()
    if "Id" in numeric_cols:
        numeric_cols.remove("Id")

    result = []

    for col in numeric_cols:
        groupWithOutMissing = useData.loc[useData[m].notna(), col].dropna()
        groupWithMissing = useData.loc[useData[m].isna(), col].dropna()

        if len(groupWithOutMissing) > 0 and len(groupWithMissing) > 0:
            stat, p_value = mannwhitneyu(
                groupWithMissing, groupWithOutMissing, alternative="two-sided"
            )
            result.append(
                {
                    "name_feature": col,
                    "value_mann_whitney_u": stat,
                    "p_value": p_value,
                    "evidence_MAR": p_value < 0.05,
                }
            )
    df_res = pd.DataFrame(result).sort_values(by="p_value")
    df_res['p_value'] = df_res['p_value'].round(5)
    if onlyTrue:
        df_res = df_res[df_res['evidence_MAR']]
    return df_res