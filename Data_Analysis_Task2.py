import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv("/content/INDIAvi - INDIAvi.csv")
data
data.shape
data.head()
data.info()
#cleaning the data
data.isnull().sum()
data.dropna(inplace=True)
data
data['tags'].unique()
data.columns
# describe() method returns description of the data in the DataFrame (i.e. count, mean, std, etc)
data.describe()
# Relationship analysis
corelation = data.corr()
sns.heatmap(corelation,xticklabels=corelation.columns, yticklabels=corelation.columns,annot=True)
sns.pairplot(data)
sns.relplot(x='category_id', y= 'views',hue='comments_disabled',data=data)
sns.distplot(data['views'])
sns.distplot(data['likes'])
sns.distplot(data['category_id'])
sns.catplot(x = 'views',kind = 'box',data =data)
sns.catplot(x = 'likes',kind = 'box',data=data)
sales_gen = data.groupby(['video_id'],as_index = False)['category_id'].sum().sort_values(by ='category_id',ascending = False)
sns.barplot(x='video_id',y = 'category_id',data=sales_gen)

fig1, ax1 = plt.subplots(figsize=(12,7))
data.groupby('category_id')['views'].sum().nlargest(10).sort_values(ascending=False).plot(kind='bar')
