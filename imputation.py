import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt


df = pd.read_csv('data/titanic.csv')


#Using Iterative Imputer with BayesianRidge estimator
iterative_imputer = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan, max_iter=10, random_state=0)

#Assuming 'titanic_data' has some MNAR data
titanic_data_imputed = iterative_imputer.fit_transform(df)

#Replace original data with imputed data
titanic_data = pd.DataFrame(titanic_data_imputed, columns=df.columns)

# #Histogram of 'Age' before imputation
# plt.hist(df['Age'].dropna(), bins=20, alpha=0.5, color='blue', label='Original')
# #Histogram of 'Age' after imputation
# plt.hist(titanic_data_imputed['Age'], bins=20, alpha=0.5, color='green', label='Imputed')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.title('Comparison of Age Distribution: Original vs. Imputed')
# plt.legend()
# plt.show()