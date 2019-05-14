from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility 为了重现结果

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard


from keras.models import load_model
from keras.utils import plot_model


model = load_model('mnist.h5')  #选取自己的.h模型名称

(X_train, y_train), (X_test, y_test) = mnist.load_data()

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))

