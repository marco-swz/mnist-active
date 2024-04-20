from typing import Annotated, Literal
from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
from keras import Model
from keras.callbacks import History
from keras.utils import to_categorical

type X = Annotated[NDArray[np.float64], Literal["N", 28, 28]]
type Y = Annotated[NDArray[np.int32], Literal["N"]]

class ActiveModel(ABC):
    @abstractmethod
    def fit(self, x_init: X, y_init: Y, x: X, y: Y):
        pass

    @abstractmethod
    def predict(self, x: X) -> NDArray:
        pass

class KerasWrapper:
    '''A wrapper class to use Keras models within the scikit-learn ecosystem'''
    batch_size: int
    model: Model
    num_epochs: int
    history: History
    callbacks: list

    def __init__(self, model, batch_size, num_epochs, callbacks=[]):
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

