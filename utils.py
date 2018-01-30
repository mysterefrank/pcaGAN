import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

def pca_reconstruct(pca, dat0, n_components):
    # Make sure we don't ask for more components than we have
    # assert n_components < 51

    # Halfway through just start training on the real dataset
    if n_components > 300:
        return dat0

    # Grab the principle components (forward pass through encoder)
    X_train_pca = pca.transform(dat0)

    # Only use the first n principle components
    X_train_pca[:, n_components:] = 0

    # Project back to original space (pass components through decoder)
    X_projected = pca.inverse_transform(X_train_pca)
    
    X_projected = (X_projected-np.min(X_projected))/(np.max(X_projected)-np.min(X_projected))
    # train_set = mnist.train.images
    X_projected = (X_projected - 0.5) / 0.5  # normalization; range: -1 ~ 1

    return X_projected


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)
