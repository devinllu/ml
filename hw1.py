from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd
import numpy as np

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


data = load_wine()

wine_data = sklearn_to_df(data)

print(wine_data.head())