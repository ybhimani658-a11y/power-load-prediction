import pandas as pd
import numpy as n
import dtale as d
import sweetviz as s
import matplotlib.pyplot as pt
import seaborn as sns

df=pd.read_csv("Z:\\Ml\\Power_load_Pridiction_552_576\\Data Set\\dataset.csv")
df.info()
#Histogram

pt.figure()
ex=df.Power_Consumption
sns.histplot(ex,kde=True)
pt.show()

#Dtale 

d=d.show(df)
d.open_browser()  

#sweetwize data

r=s.analyze(df)
r.show_html()

#scatter chart

pt.scatter(df["Power_Consumption"],df["Voltage (V)"])
pt.show()

#Box plot

pt.figure(figsize=(15,8))
numeric_df=df.select_dtypes(include=['number'])
for i,col in enumerate(numeric_df.columns):
    pt.subplot(2,5,i+1)
    pt.boxplot(numeric_df[col])
    pt.title(col)
pt.tight_layout()
pt.show()

