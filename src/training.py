from numpy.typing import NDArray
import numpy as np
import keras
import matplotlib.pyplot as plt
from svm_model import ActiveSVM
from active_model import ActiveModel


def load_dataset() -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test])


def train_model(model: ActiveModel, x: NDArray, y: NDArray):
    model.fit(x[:10], y[:10], x[10:], y[10:])

def plot_training(val_acc, acc=None, title=''):
    '''Plots the recorded training and validation accuracies during training'''

    plt.figure()
    if acc is not None:
        plt.plot(acc, label="Training")
    plt.plot(val_acc, label="Validation")
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    x, y = load_dataset()
    train_model(ActiveSVM(), x, y)

