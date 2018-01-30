import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# For some reason this doesn't work with sklearn?
# from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import fetch_mldata
import math, pdb
from sklearn.decomposition import PCA
from tflearn.datasets import cifar10
from tflearn.data_utils import shuffle, to_categorical

n_pcs = 300
def pca_reconstruct(pca, dat0, n_components):

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
    # X_projected = (X_projected - 0.5) / 0.5  # normalization; range: -1 ~ 1

    return X_projected


def show_pca(num_pcs, path = 'pca.png', original=False):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    if not original:
	    pca_samples = pca_reconstruct(pca, fixed_sample_batch, num_pcs)
    else:
    	pca_samples = fixed_sample_batch
    # Remove tiks
    pca_samples =  (pca_samples - np.max(pca_samples))/-np.ptp(pca_samples)
    # c = (255*(a - np.max(a))/-np.ptp(a)).astype(int)
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    # Fill images
    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        if dset == 'MNIST':
        	ax[i, j].imshow(np.reshape(pca_samples[k], (28, 28)), cmap='gray')
        else:
	    	ax[i, j].imshow(np.reshape(pca_samples[k], (32, 32, 3)))
    label = 'Principle Component {0}'.format(num_pcs)
    fig.text(0.5, 0.004, label, ha='center')
    plt.show()
    plt.savefig(path)


# load MNIST
dset = 'CIFAR'

if dset == 'MNIST':
	mnist = fetch_mldata('MNIST original')
	train_set = mnist.data[:].astype(np.float32)
if dset == 'CIFAR':
	(train_set, Y), _ = cifar10.load_data()
	train_set, Y = shuffle(train_set, Y)
	train_set = np.reshape(train_set, (-1, 32*32*3))

# Mix 'er up
np.random.shuffle(train_set)
# Normalize 0,1
# train_set = (train_set-np.min(train_set))/(np.max(train_set)-np.min(train_set))
# train_set = mnist.train.images
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1
# train_set = (train_set - np.mean(train_set)) / np.std(train_set)

# Samples to reconstruct in pca visualization
fixed_sample_batch = train_set[:25]

# Fit PCA on MNIST
pca = PCA(n_components=n_pcs)
pca.fit(train_set)

if not os.path.isdir('PCA_Reconstruction'):
    os.mkdir('PCA_Reconstruction')
if not os.path.isdir('PCA_Reconstruction/' + dset):
    os.mkdir('PCA_Reconstruction/' + dset)
if not os.path.isdir('PCA_Reconstruction/' + dset + '/{0}_PCs'.format(n_pcs)):
    os.mkdir('PCA_Reconstruction/' + dset + '/{0}_PCs'.format(n_pcs))
if not os.path.isdir('PCA_Reconstruction/' + dset):
    os.mkdir('PCA_Reconstruction/' + dset)

# Save target
show_pca(0, 'PCA_Reconstruction/' + dset + '/' + str(n_pcs) +  '_PCs/target', True)
pdb.set_trace()
for i in range(n_pcs):
	path = 'PCA_Reconstruction/' + dset + '/' + str(n_pcs) +  '_PCs/Principle_Component_' + str(i + 1) + '.png'
	show_pca(i, path)
