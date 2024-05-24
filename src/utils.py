from typing import Annotated, Literal, Optional
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import keras
from statsmodels.stats.proportion import proportion_confint
from keras.utils import to_categorical
from keras.callbacks import History

type X = Annotated[NDArray[np.float64], Literal["N", 28, 28]]
type Y = Annotated[NDArray[np.int32], Literal["N"]]

class Data:
    '''
    A wrapper class for the dataset. 
    It provides utility functions and ensures, only the allowed amount of labels are accessed.
    '''
    # All images
    x: X
    # All labels
    _y: Y
    # The indices of `x` which were not labeled
    idxs_unlabeled: NDArray[np.int32]
    # The indices of `x` which were already labeled
    idxs_labeled: NDArray[np.int32]
    # Maximum number of labeled data
    num_labels_max: int

    def __init__(self, x: Optional[X]=None, y: Optional[Y]=None):
        '''
        If either `x` or `y` are `None`, the full MNIST dataset is downloaded from the web.
        @param x: The images of the MNIST dataset
        @param y: The labels corresponding to the images
        @raises AssertionError: If `x` and `y` have different length
        '''
        if x is None or y is None:
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            x = np.concatenate([x_train, x_test])
            y = np.concatenate([y_train, y_test])

        assert len(x) == len(y), '`x` and `y` need to be the same length!'

        self.x = x
        self._y = y
        self.idxs_labeled = np.array([], dtype=np.int32);
        self.idxs_unlabeled = np.arange(len(x), dtype=np.int32)
        self.num_labels_max = 500

    def get_labels_for_indices(self, idxs: NDArray) -> Y:
        # TODO(marco): Ensure test data cannot be accessed
        self.idxs_labeled = np.unique(np.concatenate([self.idxs_labeled, idxs]))
        self.idxs_unlabeled = np.setdiff1d(self.idxs_unlabeled, idxs)

        assert len(self.idxs_labeled) <= self.num_labels_max, f'Nope, you only get {self.num_labels_max} labels!'
        return self._y[idxs]

    def get_test_data(self, test_ratio: float):
        idxs_all = np.random.permutation(np.arange(len(self.x)))
        idxs_train = idxs_all[:int(test_ratio*self.num_labels_max)]

        x_test = self.x[idxs_train]
        y_test = self.get_labels_for_indices(idxs_train)
        return x_test, y_test

    def reset(self):
        self.idxs_labeled = np.array([]);
        self.idxs_unlabeled = np.arange(len(self.x))

class ActiveModel(ABC):
    @abstractmethod
    def fit(self, data: Data):
        pass

    @abstractmethod
    def predict(self, x: X) -> NDArray:
        pass

class KerasWrapper:
    '''A wrapper class to use Keras models within the scikit-learn ecosystem'''
    batch_size: int
    model: keras.Model
    num_epochs: int
    history: History
    callbacks: list

    def __init__(self, model: keras.Model, batch_size, num_epochs, callbacks=[]):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.callbacks = callbacks

    def fit(self, X, y, X_val, y_val):
        self.history = self.model.fit(
            X.astype(np.float32), to_categorical(y, 10), 
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_data=(X_val.astype(np.float32), to_categorical(y_val, 10)),
            callbacks=self.callbacks,
            verbose=1
        )

    def predict(self, X):
        pred_probs = self.model.predict(X.astype(np.float32), verbose=0)
        return np.argmax(pred_probs, 1)

def eval_model(model: ActiveModel, x: X, y: Y) -> tuple[float, tuple[float, float]]:
    '''Evaluates the provided model on the provided data.'''
    predictions = model.predict(x)
    num_correct = sum(predictions == y)

    conf_int = proportion_confint(
        num_correct,
        len(y),
        alpha=0.1, 
        method="beta"
    )
    accuracy = num_correct/len(predictions)
    print('accuracy:', accuracy)
    print('conf int:', conf_int)
    return accuracy, conf_int # pyright: ignore

def test_data_class():
    data = Data(
        np.array([np.ones((28, 28))*i for i in range(510)]),
        np.arange(510),
    )

    # Getting test data
    x_test, y_test = data.get_test_data(0.5)
    assert len(x_test) == int(data.num_labels_max/2) and len(y_test) == int(data.num_labels_max/2)
    assert np.all(np.isin(data._y[data.idxs_labeled], y_test))

    # Labeling a datapoint
    idx = data.idxs_unlabeled[:1]
    y = data.get_labels_for_indices(idx)
    assert data._y[idx] == y
    assert not np.isin(idx, data.idxs_unlabeled)

    # Labeling everything possible
    data.get_labels_for_indices(data.idxs_unlabeled[:249])
    assert len(data.idxs_labeled) == 500

    # Getting the same label again
    data.get_labels_for_indices(data.idxs_labeled[:1])

    # Requesting more than allowed
    is_error = False
    try:
        data.get_labels_for_indices(data.idxs_unlabeled[:1])
    except AssertionError:
        is_error = True
    assert is_error

    print('All tests passed')

if __name__ == "__main__":
    test_data_class()
    
