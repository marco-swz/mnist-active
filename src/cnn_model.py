from numpy.typing import NDArray
import numpy as np
from scipy.stats import entropy

from active_model import ActiveModel, X, Y, Data
from keras import layers, Sequential, Input, Model, utils

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
        batch_size = 128
        search_size = 1000

        x = data.x
        x = x.astype("float32") / 255
        x = np.expand_dims(x, -1)

        idxs_lbl = np.random.choice(len(x), batch_size, replace=False)

        for _ in range(100):
            y = data.get_labels(idxs_lbl)
            y = utils.to_categorical(y, 10)

            self.model.fit(x[idxs_lbl], y, batch_size=batch_size, epochs=1)

            idxs_search = np.random.choice(len(data.idxs), search_size, replace=False)
            pred_probs = self.model.predict(x[idxs_search])
            entropies = entropy(pred_probs)

            idxs_max = np.argsort(entropies)[-batch_size:]
            idxs_lbl = idxs_search[idxs_max]

    def predict(self, x: X) -> NDArray:
        x = x.astype("float32") / 255
        x = np.expand_dims(x, -1)
        preds = np.argmax(self.model.predict(x), 1)
        print(preds)
        return preds
