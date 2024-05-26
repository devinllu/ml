from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
iris_data = pd.DataFrame(iris.data, columns=iris.feature_names)

# print(iris_data.head())
# print(iris_data.info())
scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
iris_scaled = scaler.fit_transform(iris_data)
iris_min_max_scaled = min_max_scaler.fit_transform(iris.data)
feature = 'sepal length (cm)'

def show_histogram(): # creates a histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(iris_data[feature], kde=True, color='blue')
    plt.title(f'Original {feature}')
            
    plt.subplot(1, 3, 2)
    sns.histplot(iris_scaled[:, 0], kde=True, color='green')  # Adjust index accordingly
    plt.title(f'Standardized {feature}')

    plt.subplot(1, 3, 3)
    sns.histplot(iris_min_max_scaled[:, 0], kde=True, color='red')
    plt.title(f'Min Max Standardized {feature}')
    plt.show()

def show_box_plot():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    sns.boxplot(data=iris_data, y=feature)
    plt.title(f'Original {feature}') 
            
    plt.subplot(1, 3, 2)
    sns.boxplot(y=iris_scaled[:, 0])  # Adjust index accordingly
    plt.title(f'Standardized {feature}')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=iris_min_max_scaled[:, 0])  # Adjust index accordingly
    plt.title(f'Standardized {feature}')
    plt.show() 

def show_scatter_plot():
    plt.scatter(iris_data['sepal length (cm)'], iris_data['sepal width (cm)'], alpha=0.5, label='Original')
    plt.scatter(iris_scaled[:, 0], iris_scaled[:, 1], alpha=0.5, label='Standardized')  # Adjust indices accordingly
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Sepal Length vs. Width: Original vs. Standardized')
    plt.legend()
    plt.show() 

# show_histogram()
# show_box_plot()
show_scatter_plot