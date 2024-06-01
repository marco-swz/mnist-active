from __future__ import annotations
from typing import override

from numpy.typing import NDArray, ArrayLike
import numpy as np
from scipy.stats import entropy
from keras import layers, Sequential, Input, Model, utils
import skopt

from utils import eval_model, ActiveModel, optimize_model, MAX_NUM_LABELED, split_data, load_data, NUM_TEST

class ActiveCNN(ActiveModel):
    model: Model
    params: dict
    verbose: bool
    num_train: int

    def __init__(self, **params):
        params["size_cnn"] = params.get("size_cnn", 32)
        params["size_dense"] = params.get("size_dense", 10)
        params["num_dense"] = params.get("num_dense", 1)
        #self.num_train = MAX_NUM_LABELED - DataPoint._num_labeled
        self.num_train = MAX_NUM_LABELED - NUM_TEST

        self.verbose = params.get("verbose", False)

        self.params = params

        input_shape = (28, 28, 1)
        net = [
            Input(shape=input_shape),
            layers.Conv2D(self.params["size_cnn"], kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
        ]

        for _ in range(self.params["num_dense"]):
            net.append(layers.Dense(self.params["size_dense"], activation="relu"))

        net.append(layers.Dense(10, activation="softmax"))

        self.model = Sequential(net)
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

        num_iterations = int(self.num_train / batch_size)
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

            self.model.fit(x, y, batch_size=batch_size, epochs=50, verbose=0)

        #print(DataPoint._num_labeled)

    def predict(self, x: NDArray) -> NDArray:
        x = x.astype("float32") / 255
        x = np.expand_dims(x, -1)
        preds = np.argmax(self.model.predict(x, verbose=0), 1)
        #print(preds)
        return preds

    def get_params(self, deep: bool=True):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self

if __name__ == "__main__":
    data = load_data()
    #data_train, data_test = split_data(data, 0.2)
    #x_test = np.array([d.x for d in data_test])
    #y_test = np.array([d.get_label() for d in data_test])

    model = ActiveCNN()
    optimize_model(
        model=model,
        opt_params=dict(
            size_cnn=skopt.space.Integer(16, 128),
            size_dense=skopt.space.Integer(10, 128),
            num_dense=skopt.space.Integer(1, 10),
        ),
        data=data,
    )

    #model.fit(data_train)
    #eval_model(model, x_test, y_test)
