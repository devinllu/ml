'''
Homework #1: Exploring the Wine Dataset

1st iteration: ~68% accuracy with moderate f1-score and cross validation scores
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

wine_data = pd.read_csv('data/wine_quality.csv')

def train_model():
    wine_data.fillna(wine_data.mean(), inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(wine_data.drop('quality', axis=1))

    x, y = scaled, wine_data['quality']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(x_train, y_train)

    predictions = rfc.predict(x_test)
    print(f'Accuracy of RFC model: {accuracy_score(y_test, predictions)}')

    # confusion matrix points out true pos/neg along the diagonal, and false pos/neg elsewhere
    print(f'Confusion Matrix:\n {confusion_matrix(y_test, predictions)}')

    # used to evaluate the performance of your model
    print(f'Classification Report:\n {classification_report(y_test, predictions, zero_division=0)}')

    cross_validate(rfc, x, y)

def cross_validate(model, x, y):
    scores = cross_val_score(model, x, y, cv=5)
    rounded = [round(score, 3) for score in scores]

    print(f'Cross Validation Scores: {rounded}')


def print_stats():

    # print(wine_data.head())
    # print(wine_data.tail())
    print(f'dataframe info: {wine_data.info()}')

    print(f'wine quality mean: {round(np.mean(wine_data["quality"]), 3)}')
    print(f'wine quality variance: {round(np.var(wine_data["quality"]), 3)}')
    print(f'wine quality standard deviation: {round(np.std(wine_data["quality"], ddof=1), 3)}')
    print(f'max value: {np.max(wine_data["quality"])}')
    print(f'min value: {np.min(wine_data["quality"])}')

    # Calculate quartiles
    Q1 = np.quantile(wine_data["quality"],0.25)
    Q3 = np.quantile(wine_data["quality"],0.75)

    # Calculate the Interquartile Range
    IQR = Q3 - Q1

    print("Q1 (25th percentile): ", Q1)
    print("Q3 (75th percentile): ", Q3)
    print("Interquartile Range: ", IQR)

def plot_heatmap():
    '''
    creates a correlation matrix between each pairs of variables. ranges from -1 (neg-corr) to +1 (pos-corr)
    0 indicates no correlation
    '''
    correlation_matrix = wine_data.corr()

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))  # Set the size of the figure
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    plt.title('Correlation Matrix for Wine Dataset')
    plt.show()

# print_stats()
# plot_heatmap()
train_model()



