import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA()

principal_components = pca.fit_transform(X_scaled)

print(pca.get_feature_names_out())

df = pd.DataFrame(data=principal_components)
print(df.head())

final_df = pd.concat([df,iris['species']],axis=1)

exp_var = pca.explained_variance_ratio_*100

plt.plot(
    range(1,5),
    exp_var,   
)

plt.show()

plt.scatter(
    x='pca0',
    y='pca1',
    data = final_df
)
plt.show()