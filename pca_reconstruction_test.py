from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
import numpy as np
from numpy.testing import assert_array_almost_equal
import pdb
import matplotlib.pyplot as plt
# import tensorflow as tf

mnist = fetch_mldata('MNIST original')
dat0 = mnist.data[:].astype(np.float32)
# normalize data
dat0 = (dat0 - np.mean(dat0)) / np.std(dat0)

pca = PCA(n_components=50)
pca.fit(dat0)

# Grab the principle components (forward pass through encoder)
X_train_pca = pca.transform(dat0)

# Only use the first n principle components
X_train_pca[:, 49:] = 0

# Project back to original space (pass components through decoder)
X_projected = pca.inverse_transform(X_train_pca)

# Calculate loss if you want
# loss = ((dat0 - X_projected) ** 2).mean()

# Visualize some 
fig, axes = plt.subplots(nrows=4, ncols=4)
# merge 2d list
axes = axes.reshape(-1)
for ax, img in zip(axes, X_projected[:len(axes)]):
	ax.imshow(img.reshape(28,28))
plt.show()