from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

import pandas as pd
import numpy as np

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    return df

def prepocess_data(df):
    df.fillna(df.mean(), inplace=True)

    scaler = StandardScaler()
    return scaler.fit_transform(df.drop('quality', axis=1))

data = pd.read_csv('data/wine_quality.csv')
wine_data_scaled = prepocess_data(data)

x, y = wine_data_scaled, data['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

cross_val_scores = cross_val_score(model, x, y, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
