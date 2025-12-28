import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from IPython.display import display

def resumen_analyst(df_train, name_column = '', display_graphs=True):
    if display_graphs:
        plot_analyst(df_train, name_column)
    res = cal_quartiles(df_train[name_column])
    display(res)

def plot_analyst(df_train, name_column = ''):
    #Boxplot
    plt.figure(figsize=(30,7))
    sns.boxplot(x=df_train[name_column], color='lightblue')
    plt.title('Boxplot of YearBuilt');

    #Histogram
    plt.figure(figsize=(10,6))
    sns.histplot(df_train[name_column], bins=50, kde=True, color='lightgreen')
    plt.title('Distribution of YearBuilt');
    plt.xlabel(name_column)
    plt.ylabel('Frequency');


    #Q-Q plot
    plt.figure(figsize=(6,6))
    stats.probplot(df_train[name_column], dist="norm", plot=plt)
    plt.title('Q-Q Plot of YearBuilt');
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles');
    plt.grid(True);
    print('---'*20)
    
    
def cal_quartiles(df_in):
    Q1 = np.percentile(df_in, 25)
    Q2 = np.percentile(df_in, 50)
    Q3 = np.percentile(df_in, 75)
    RIC = Q3 - Q1
    W1 = Q1 - 1.5 * RIC
    W1 = np.max(df_in[df_in >= W1]) if np.any(df_in >= W1) else np.min(df_in)
    W2 = Q1 + 1.5 * RIC
    W2 = np.min(df_in[df_in <= W2]) if np.any(df_in <= W2) else np.max(df_in)
    
    format_result = lambda x: f'{x:.2f}'
    
    return {
        "quartile_25": format_result(Q1),#Primer cuartil
        "quartile_50": format_result(Q2),#Mediana
        "quartile_75": format_result(Q3),#Tercer cuartil
        "RIC": format_result(RIC), #Rango Intercuartílico
        "whisker_lower": format_result(W1),#Límite inferior de los bigotes
        "whisker_upper": format_result(W2),#Límite superior de los bigotes
    }