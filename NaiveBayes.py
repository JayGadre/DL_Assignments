from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

Nb = GaussianNB()

Nb.fit(X_train,y_train)

y_pred = Nb.predict(X_test)

print(classification_report(y_test,y_pred))




