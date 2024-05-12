import numpy as np
import os
from urllib.request import urlretrieve
import gzip


def get_mnist():
    path = os.path.join(os.curdir, 'mnist')
    os.makedirs(path, exist_ok=True)

    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for file in files:
        if file not in os.listdir(path):
            urlretrieve('http://yann.lecun.com/exdb/mnist/' + file, os.path.join(path, file))

    return [
        np.frombuffer(gzip.open(os.path.join(path, files[0])).read(), 'B', offset=16).reshape(-1, 28, 28).astype('float32') / 255.,
        np.frombuffer(gzip.open(os.path.join(path, files[1])).read(), 'B', offset=8),
        np.frombuffer(gzip.open(os.path.join(path, files[2])).read(), 'B', offset=16).reshape(-1, 28, 28).astype('float32') / 255.,
        np.frombuffer(gzip.open(os.path.join(path, files[3])).read(), 'B', offset=8)
    ]
