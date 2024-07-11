import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Display first few rows, shape and check for null values
print(df.head())
print(df.shape)
print(df.isnull().any())

# Rename columns
df.rename(columns={'Spending Score (1-100)': 'Spending Score', 'Annual Income (k$)': 'Annual Income', 'Genre':'Gender'}, inplace=True)

# Separate the dataset into features (X) and labels (y)
y = df['Spending Score']
X = df.drop(columns='Spending Score')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Encode the categorical column 'Gender'
labelencoder = LabelEncoder()
X_train['Gender'] = labelencoder.fit_transform(X_train['Gender'])
X_test['Gender'] = labelencoder.fit_transform(X_test['Gender'])

# Standardize the dataset
std_scaler = StandardScaler()
scaled_X_train = pd.DataFrame(std_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
scaled_X_test = pd.DataFrame(std_scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

scaled_X_train['Gender'] = labelencoder.fit_transform(X_train['Gender'])
scaled_X_test['Gender'] = labelencoder.fit_transform(X_test['Gender'])

# Support Vector Regression Model
svm_model = svm.SVR()
svm_model.fit(scaled_X_train, y_train)
svm_prediction = svm_model.predict(scaled_X_test)
svm_mae = mean_absolute_error(y_test, svm_prediction)
print(f'SVM MAE: {svm_mae}')

# Decision Tree Regressor Model
dt_model = DecisionTreeRegressor()
dt_model.fit(scaled_X_train, y_train)
dt_prediction = dt_model.predict(scaled_X_test)
dt_mae = mean_absolute_error(y_test, dt_prediction)
print(f'Decision Tree MAE: {dt_mae}')

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(scaled_X_train, y_train)
lr_prediction = lr_model.predict(scaled_X_test)
lr_mae = mean_absolute_error(y_test, lr_prediction)
print(f'Linear Regression MAE: {lr_mae}')

