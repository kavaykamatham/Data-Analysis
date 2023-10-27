import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
data = pd.read_csv('/content/DATA - 3 - DATA - 3 (7).csv')
data
data.info()
data.head()
data.tail()
data.columns
data.shape
data.describe()
data.dropna(inplace=True)
data
data.duplicated()
data.drop_duplicates()
data.drop_duplicates(inplace=True)
data
print(data.isnull().sum())
from sklearn.model_selection import train_test_split
x_train ,x_test , y_train , y_test = train_test_split(x_data,y_data,test_size = 0.4, random_state =10)
sns.relplot(x = 'responseID',y ='participantID',data=data)
sns.relplot(x = 'age',y ='education',data=data)
sns.relplot(x = 'nativeLanguage',y ='city',data=data)
sns.relplot(x='education',y='age',hue = 'gender',data=data)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data.info()
print(data.dtypes)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in data.columns:
  if data[col].dtype == 'object':
    data[col] = encoder.fit_transform(data[col])
train = data.drop(['nativeLanguage','gender','city',],axis=1)
test = data['nativeLanguage']
x_train ,x_test , y_train , y_test = train_test_split(train,test,test_size = 0.4, random_state =10)
regr = LinearRegression()
regr.fit(x_train,y_train)
pred = regr.predict(x_test)
pred
regr.score(x_test,y_test)
from sklearn import metrics
from sklearn.metrics import accuracy_score
#LogisticRegression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
prediction_on_training_data = model.predict(x_train)
accuracy_on_training_data = accuracy_score(y_train,prediction_on_training_data)
print('Accuracy on training data:',accuracy_on_training_data)
prediction_on_test_data = model.predict(x_test)
accuracy_on_test_data = accuracy_score(y_test,prediction_on_test_data)
print('Accuracy on test data:',accuracy_on_test_data)
#RandomForest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
X = data.drop('city', axis=1)
y = data['city']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{classification_report}')
print(f'Accuracy: {accuracy}')