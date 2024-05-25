from numpy.typing import NDArray
import numpy as np
from scipy.stats import entropy
from keras import layers, Sequential, Input, Model, utils

from utils import Data, X, Y, eval_model, ActiveModel

class ActiveCNN(ActiveModel):
    model: Model
    x: X|None
    y: Y|None

    def __init__(self):
        input_shape = (28, 28, 1)
        self.model = Sequential([
            Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
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

    def fit(self, data: Data):
        batch_size = 25
        search_size = 1000

        idxs_to_label = np.random.choice(data.unlabeled_indices(), batch_size, replace=False)

        num_iterations = int(data.num_unlabeled_remaining() / batch_size)
        for _ in range(num_iterations):
            x = data.features(idxs_to_label)
            x = x.astype("float32") / 255
            x = np.expand_dims(x, -1)

            y = data.request_labels(idxs_to_label)
            y = utils.to_categorical(y, 10)

            self.model.fit(x, y, batch_size=batch_size, epochs=50, verbose=1)

            idxs_search = np.random.choice(data.unlabeled_indices(), search_size, replace=False)
            x_search = data.features(idxs_search)
            x_search = x_search.astype("float32") / 255
            x_search = np.expand_dims(x_search, -1)
            pred_probs = self.model.predict(x_search, verbose=0)

            entropies = entropy(pred_probs, axis=1)
            idxs_max = np.argsort(entropies)[-batch_size:]
            idxs_to_label = idxs_search[idxs_max]

        print(data.num_unlabeled_remaining())

    def predict(self, x: X) -> NDArray:
        x = x.astype("float32") / 255
        x = np.expand_dims(x, -1)
        preds = np.argmax(self.model.predict(x), 1)
        print(preds)
        return preds

if __name__ == "__main__":
    data = Data()
    x_test, y_test = data.get_test_data(test_ratio=0.2)
    model = ActiveCNN()
    model.fit(data)
    eval_model(model, x_test, y_test)
