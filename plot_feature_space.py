
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

h5file = h5py.File("C:\\Users\\arkaniva\\Downloads\\AI-vs.-Human-Generated-Images-Kaggle-Competition\\src\\AI-vs.-Human-Generated-Images-Kaggle-Competition\\efficientNetB7_features.h5",'r')
features = h5file['features'][:]
labels = h5file['labels'][:]
h5file.close()

# apply pca and vizualize
pca = PCA(n_components=3)
features_pca = pca.fit_transform(features)
    
# visualize
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features_pca[:, 0], features_pca[:, 1], features_pca[:, 2], 
            c=labels, alpha=0.5)
plt.show()

