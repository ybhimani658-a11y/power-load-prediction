# Flight Price Prediction Project

# Import the Python Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Data Collection

df = pd.read_csv('Z:\\Ml\\Power_load_Pridiction_552_576\\Data Set\\dataset.csv')
print(df)



# Data Cleaning & Data cleaning

print("missing values: ",df.isnull().sum())
print("duplicate values: ",df.duplicated().sum())
print("unique values: ",df.nunique())

print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
qty = df.select_dtypes(include=['int64','float']).columns
print(qty)

qlty = df.select_dtypes(include=['object','category']).columns
print(qlty)



print("missing values: ",df.isnull().sum())
print("duplicate values: ",df.duplicated().sum())
print("unique values: ",df.nunique())



# Handle the outliers of Duration time

Q1=df['Power_Consumption'].quantile(0.25)
Q3=df['Power_Consumption'].quantile(0.75)

IQR = Q3 - Q1

# set the limit

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Power_Consumption'] >= lower_bound) & (df['Power_Consumption'] <= upper_bound)]


# Exploratory Data Analysis 



# Histogram

plt.figure(figsize=(15,8))
plt.hist(df['Power_Consumption'], bins=20)
plt.xlabel('Power Consumption')
plt.ylabel('Frequency')
plt.title('Distribution of Power Consumption')
plt.savefig('Histogram.png')
plt.show()


# Boxplot

plt.figure(figsize=(15,8))
plt.boxplot(df['Power_Consumption'])
plt.ylabel('Power Consumption')
plt.title('Box Plot of Power Consumption')
plt.savefig('Box_Plot_After.png')
plt.show()



# Coorelation Matrix

import seaborn as sns

numeric_df = df.select_dtypes(include='number')

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('Heatmap.png')
plt.show()




# Automated EDA

import sweetviz as sv

report =  sv.analyze(df)
report.show_html()


# Feature SelecƟon 


# Dependent & Independent variables

X = df[['Power_Consumption', 'Voltage (V)', 'Current (A)', 'Power Factor', 'Voltage Fluctuation (%)',
       'Humidity (%)', 'Reactive Power (kVAR)']]

y = df['Voltage (V)']

print(X.shape)
print(y.shape)


# Train-Test Split 


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




# Build ML Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

print('Model build successfully!')




# Predct the X_test

y_pred = model.predict(X_test)
print(y_pred)



# Model Evaluation

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse**0.5
r2 = r2_score(y_test,y_pred)

print("MAE: ",mae)
print("MSE: ",mse)
print("RMSE: ",rmse)
print("R2: ",r2)



# Improve Model



# Cross Validation

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("Cross Validation R² Scores:")
print(cv_scores)

print("Mean R²:", cv_scores.mean())



# Ridge Model

from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# analyze ridge


from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score
)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Ridge Regression")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)