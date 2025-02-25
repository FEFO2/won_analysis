import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------------------------------------------------------------
def plot_numerical_data(dataframe):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_columns:
        fig, axis = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={'height_ratios': [6, 1]})

        # Calculate mean, median, and standard deviation
        mean_val = np.mean(dataframe[column])
        median_val = np.median(dataframe[column])
        std_dev = np.std(dataframe[column])

        # Create a multiple subplots with histograms and box plots
        sns.histplot(ax=axis[0], data=dataframe, kde=True, x=column).set(xlabel=None)
        axis[0].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[0].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[0].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1, label='Standard Deviation')
        axis[0].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)

        sns.boxplot(ax=axis[1], data=dataframe, x=column, width=0.6).set(xlabel=None)
        axis[1].axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label='Mean')
        axis[1].axvline(median_val, color='orange', linestyle='dashed', linewidth=1, label='Median')
        axis[1].axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=1)
        axis[1].axvline(mean_val - std_dev, color='green', linestyle='dashed', linewidth=1)

        axis[0].legend()

        fig.suptitle(column)

        # Adjust the layout
        plt.tight_layout()

        # Show the plot
        plt.show()

# -------------------------------------------------------------------------------
def plot_scatter_heatmaps(dataframe, target_variable):
    numeric_variables = dataframe.select_dtypes(include=['float64', 'int64']).columns
    num_cols = 2
    num_rows = len(numeric_variables) - 1

    fig, axis = plt.subplots(num_rows, num_cols, figsize=(13, 5 * num_rows))

    for i, x_variable in enumerate(numeric_variables):
        # Evitar plotear la variable target
        if x_variable == target_variable:
            continue

        # Gráfico de dispersión
        sns.regplot(ax=axis[i, 0], data=dataframe, x=x_variable, y=target_variable)
        axis[i, 0].set_title(f'Regplot: {x_variable} vs {target_variable}')

        # Mapa de calor
        sns.heatmap(dataframe[[x_variable, target_variable]].corr(), annot=True, fmt=".2f", ax=axis[i, 1])
        axis[i, 1].set_title(f'Heatmap: {x_variable} vs {target_variable}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar la posición del título
    plt.show()

# -------------------------------------------------------------------------------
def general_heatmap(dataframe,target_variable):
    #Reorder the df with the target column first
    columns = [target_variable] + [col for col in dataframe if col != target_variable]
    reordered_df = dataframe[columns]
    #Create general heatmap
    fig, axis = plt.subplots(figsize = (10, 7))
    sns.heatmap(reordered_df.corr(), annot = True, fmt = ".2f", cbar = False)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------

def early_split(dataframe, target):
    # Select the numeric variables
    numeric_variables = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numeric_variables = numeric_variables[numeric_variables != target]
    # Separate the target and the predictors
    x = dataframe.drop(target, axis=1)[numeric_variables]
    y = dataframe[target]
    # Divide the dataset into train / test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    x_train[target] = list(y_train)
    x_test[target] = list(y_test)
    # # Create the csv documents for store
    # dataframe.to_csv('early_total_data.csv',index = False)
    # x_train.to_csv('early_x_train.csv', index = False )
    # x_test.to_csv('early_x_test.csv', index = False )

# -------------------------------------------------------------------------------

def outlier_analysis(dataframe,target):
    numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numerical_columns = numerical_columns[numerical_columns != target]
    for column in numerical_columns:
        fig, axis = plt.subplots(figsize=(8, 1.2))
        sns.boxplot(ax=axis, data=dataframe, x=column, width=0.3).set(xlabel=None)
        fig.suptitle(column)
        plt.tight_layout()
        plt.show()
    # Return the describe dataframe    
    return dataframe.describe().T

# -------------------------------------------------------------------------------

def min_max_records(dataframe, target, record_num):
    for x in dataframe.columns:
        if x == target:
            continue
        # VARIABLES FOR PLOTTING
        min_values = np.sort(dataframe[x].unique())[:record_num]
        max_values = np.sort(dataframe[x].unique())[-record_num:]
        min_records = [(dataframe[x] == value).sum() for value in min_values]
        max_records = [(dataframe[x] == value).sum() for value in max_values]

        # PLOT WITH 2 SUBPLOTS
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # MIN VALUE PLOT
        ax1.plot(min_values, min_records, linestyle='-', color='b', label='Occurrences', marker='o')
        ax1.set_xlabel(f'{x}')
        ax1.set_ylabel('Occurrences')
        ax1.set_title(f'Occurrences of {x} minimum values')
        ax1.grid(True)
        ax1.legend()

        # MAX VALUE PLOT
        ax2.plot(max_values, max_records, linestyle='-', color='g', label='Occurrences', marker='o')
        ax2.set_xlabel(f'{x}')
        ax2.set_title(f'Occurrences of {x} maximum values')
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        plt.show()

# -------------------------------------------------------------------------------

def outliers_summary(dataset,outliers):
    print(f'''the rows with outliers are {len(outliers)}''')
    print(f'''the total rows are {len(dataset)}''')
    print(f'''this represents {round(len(outliers)/len(dataset),2)*100} % of the dataset''')

# -------------------------------------------------------------------------------

def dual_heatmap(dataframe1, dataframe2, target_variable):
    # Reorder the dataframes with the target column first
    columns1 = [target_variable] + [col for col in dataframe1 if col != target_variable]
    reordered_df1 = dataframe1[columns1]

    columns2 = [target_variable] + [col for col in dataframe2 if col != target_variable]
    reordered_df2 = dataframe2[columns2]
    
    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Heatmap for the first dataframe
    sns.heatmap(reordered_df1.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title(f'Correlation Heatmap: DataFrame 1')

    # Heatmap for the second dataframe
    sns.heatmap(reordered_df2.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title(f'Correlation Heatmap: DataFrame 2')

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------

def small_histogram(dataset, target):
    # Obtener la lista de todas las columnas numéricas en el DataFrame
    numeric_variables = dataset.select_dtypes(include=['float64', 'int64']).columns
    sns.set(style="whitegrid")
    # Crear subgráficos de histogramas
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
    for i, variable in enumerate(numeric_variables):
        if variable == target:
            continue
        sns.histplot(dataset[variable], bins=20, kde=True, ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f'{variable}')

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------------

def clean_split(dataframe, target):
    # Select the numeric variables
    numeric_variables = dataframe.select_dtypes(include=['float64', 'int64']).columns
    numeric_variables = numeric_variables[numeric_variables != target]
    # Separate the target and the predictors
    x = dataframe.drop(target, axis=1)[numeric_variables]
    y = dataframe[target]
    # Divide the dataset into train / test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    x_train[target] = list(y_train)
    x_test[target] = list(y_test)
    # # Create the csv documents for store
    dataframe.to_csv('clean_total_data.csv',index = False)
    x_train.to_csv('clean_x_train.csv', index = False )
    x_test.to_csv('clean_x_test.csv', index = False )

# -------------------------------------------------------------------------------

def corr_comparison(dataset,dataset2,dataset3,target):
    corr1 = dataset.corr()[target][:-1]
    corr2 = dataset2.corr()[target][:-1]
    corr3 = dataset3.corr()[target][:-1]

    comparison_df = pd.DataFrame({
        'Original': corr1,
        'Filtered': corr2,
        'Mean rep.': corr3
    })

    return comparison_df