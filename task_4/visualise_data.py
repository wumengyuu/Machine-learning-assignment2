import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


def plot_data_imputation(df_raw, df_imputed, column1, column2, column3):
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    df_raw_pca = pd.DataFrame(pca.fit_transform(df_raw.fillna(0)), columns=[column1, column2, column3])
    df_imputed_pca = pd.DataFrame(pca.transform(df_imputed), columns=[column1, column2, column3])

    # Visualization
    plt.figure(figsize=(12, 6))

    # Before Imputation
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=column1, y=column2, data=df_raw_pca, color='red', label='Before Imputation')
    plt.title('Before Imputation')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # After Imputation
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=column1, y=column2, data=df_imputed_pca, color='blue', label='After Imputation')
    plt.title('After Imputation')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()
