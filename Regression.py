from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,root_mean_squared_error
import matplotlib.pyplot as plt

diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
df['target'] = diabetes.target

X = df[['bmi']]

y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

Lr = LinearRegression()

Lr.fit(X_train,y_train)

y_pred = Lr.predict(X_test)

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,color='red')
plt.show()

X = df.iloc[:,:-1]

y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

Lr = LinearRegression()

Lr.fit(X_train,y_train)

y_pred = Lr.predict(X_test)

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(root_mean_squared_error(y_test,y_pred))

plt.scatter(y_test,y_pred)
plt.plot(y_pred,y_pred,color='red')
plt.show()



