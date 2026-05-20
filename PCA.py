import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# --- a. Pre-process the data through standardization ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- b. Perform PCA to reduce dimension ---
# We fit PCA on the scaled data
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Get the auto-generated feature names out (pca0, pca1, etc.)
feature_names = pca.get_feature_names_out()

# Create a clean DataFrame with the principal component columns
df = pd.DataFrame(data=principal_components, columns=feature_names)

# Map the numeric target classes (0, 1, 2) to actual species names for better visualization
df['species'] = [target_names[i] for i in y]

# --- c. Construct the scree plot ---
exp_var = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(7, 4))
plt.plot(range(1, 5), exp_var, marker='o', linestyle='--', color='b')
plt.title('Scree Plot (Explained Variance by Component)')
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Variance Explained (%)')
plt.xticks(range(1, 5))
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# --- d. Data visualization in lower dimensional representation ---
plt.figure(figsize=(8, 6))
# Using seaborn makes it incredibly easy to color-code by species
sns.scatterplot(
    x='pca0', 
    y='pca1', 
    hue='species', 
    data=df, 
    palette='Set1', 
    s=70
)
plt.title('PCA - Iris Dataset (Lower Dimensional Space)')
plt.xlabel(f'PCA 1 ({exp_var[0]:.1f}%)')
plt.ylabel(f'PCA 2 ({exp_var[1]:.1f}%)')
plt.legend(title='Species')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
