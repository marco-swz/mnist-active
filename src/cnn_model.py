from __future__ import annotations

from numpy.typing import NDArray, ArrayLike
import numpy as np
from scipy.stats import entropy
from keras import layers, Sequential, Input, Model, utils
import skopt
import keras

from utils import DataPoint, X, Y, eval_model, ActiveModel, optimize_model, MAX_NUM_LABELED, split_data

class ActiveCNN(ActiveModel):
    model: Model
    size_cnn: int

    def __init__(self, size_cnn: int=32):
        self.size_cnn = size_cnn
        input_shape = (28, 28, 1)
        self.model = Sequential([
            Input(shape=input_shape),
            layers.Conv2D(size_cnn, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ])
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

    def fit(self, data_unlabeled: ArrayLike):
        batch_size = 25
        search_size = 1000

        data_labeled = np.array([])
        data_unlabeled = np.array(data_unlabeled)

        num_iterations = int((MAX_NUM_LABELED - DataPoint._num_labeled) / batch_size)
        for _ in range(num_iterations):
            idxs_search = np.random.choice(len(data_unlabeled), search_size, replace=False)
            x_search = np.array([d.x for d in data_unlabeled[idxs_search]])
            x_search = x_search.astype("float32") / 255 
            x_search = np.expand_dims(x_search, -1)
            pred_probs = self.model.predict(x_search, verbose=0)

            entropies = entropy(pred_probs, axis=1)
            idxs_max = np.argsort(entropies)[-batch_size:]
            idxs_to_label = idxs_search[idxs_max]

            data_to_label = data_unlabeled[idxs_to_label]
            data_labeled = np.concatenate([data_labeled, data_to_label])
            data_unlabeled = np.delete(data_unlabeled, idxs_to_label)

            x = np.array([d.x for d in data_labeled])
            x = x.astype("float32") / 255 
            x = np.expand_dims(x, -1)

            y = np.array([d.get_label() for d in data_labeled])
            y = utils.to_categorical(y, 10)

            self.model.fit(x, y, batch_size=batch_size, epochs=50, verbose=1)

        print(DataPoint._num_labeled)

    def predict(self, x: NDArray) -> NDArray:
        x = x.astype("float32") / 255
        x = np.expand_dims(x, -1)
        preds = np.argmax(self.model.predict(x), 1)
        print(preds)
        return preds

    def get_params(self, deep: bool):
        return { "size_cnn": self.size_cnn }

if __name__ == "__main__":
    data = load_data()
    data_train, data_test = split_data(data, 0.2)
    x_test = np.array([d.x for d in data_test])
    y_test = np.array([d.get_label() for d in data_test])

    model = ActiveCNN()
    optimize_model(
        model=model,
        params={
            'size_cnn': skopt.space.Integer(16, 128)
        },
        data=data,
    )

    #model.fit(data_train)
    #eval_model(model, x_test, y_test)
