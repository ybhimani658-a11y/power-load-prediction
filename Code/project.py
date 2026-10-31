import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, mean_absolute_error  
# -------------------------------------------------- # Step 1 : Load Dataset 
df = pd.read_csv("Z:\\Ml\\Power_load_Pridiction_552_576\\Data Set\\dataset.csv")
print("Complete Dataset") 
print(df)  

# -------------------------------------------------- # Step 2 : Understand Dataset 
print("\nShape :", df.shape) 
print(df.columns)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
qty = df.select_dtypes(include=['int64','float']).columns
print(qty)
print("\nColumns :", df.columns.tolist())  
qlty = df.select_dtypes(include=['object','category']).columns
print(qlty)

print("missing values: ",df.isnull().sum())
print("duplicate values: ",df.duplicated().sum())
print("unique values: ",df.nunique())


# -------------------------------------------------- # Step 3 : Identify Features and Label 
X = df[['Power_Consumption', 'Voltage (V)', 'Current (A)', 'Power Factor', 'Voltage Fluctuation (%)',
       'Humidity (%)', 'Reactive Power (kVAR)']]      # Feature 
y = df["Power_Consumption"]             # Label  
print("\nFeature (Input)") 
print(X.head())  
print("\nLabel (Output)") 
print(y.head())  
print("\nFeature Information")
print(X.info())

print("\nUnique values")
for column in X.columns:
    print(column, ":", X[column].unique()[:20])
print("Unique years:", df["Year"].unique())
print("Number of unique Years:", df["Year"].nunique())
print("Year standard deviation:", df["Year"].std())

plt.figure(figsize=(8, 5))
plt.boxplot(df['Power_Consumption'])
plt.ylabel('Power Load Time (minutes)')
plt.title('Box Plot of Power Load Time')
plt.savefig('Box_Plot_Before.png')
plt.show()

# -------------------------------------------------- # Step 4 : Type of Machine Learning 
print("\nMachine Learning Type : Supervised Learning") 
print("Problem Type(Regression / Classification): Regression")  


# -------------------------------------------------- # Step 5 : Split Dataset 
# #train_test_split function: Used to split dataset in training and testing set. 
# #test_size: mention splitting ratio (0.20 means 20% split for testing dataset) 
# #random_state: used for fixed randomization every time when you again split data. 
# # 42 is just a starting seed which is used by random number generation method for starting.  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
print("\nTraining Records :", len(X_train)) 
print("Testing Records  :", len(X_test))
 
 
 # Step 6 : Train Model 
model = LinearRegression() 
model.fit(X_train, y_train)  

# -------------------------------------------------- # Step 7 : Prediction 
# # Prepared model is tested with testing dataset. 
y_pred = model.predict(X_test)  
print("\nActual power consumption") 
print(y_test.values) 
print("\nPredicted power consumption") 
print(y_pred)  
# -------------------------------------------------- # Step 8 : Error Calculation 
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)  
print("\nMean Absolute Error (MAE) :", mae) 
print("Mean Squared Error (MSE) :", mse)  

r2 = model.score(X_test, y_test) 
print("R² value:", r2) 


#-------------------------------------------------- # Step 9 : Predict New Student 
hours = [[0,6,4,2019,1220,1065,180]] 
prediction = model.predict(hours)  
print("\nIf your flight predicted power consumption is: ", prediction[0]) 

print("\nCorrelation with Power Consumption")
print(df.corr(numeric_only=True)["Power_Consumption"].sort_values(ascending=False))
