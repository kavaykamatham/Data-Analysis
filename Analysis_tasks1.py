#importing libraries
import pandas as pd

#importing dataset
data = pd.read_csv("//content/01.Data Cleaning and Preprocessing.csv")
print(data)
print(data.head(5))

#removing duplicates
data.dropna(inplace=True)
data
data.drop_duplicates()
data.drop_duplicates(inplace=True)
data
print(data.isnull().sum())