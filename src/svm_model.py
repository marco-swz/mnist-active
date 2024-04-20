from numpy.typing import NDArray

from sklearn.svm import SVC
from active_model import ActiveModel, X, Y

class ActiveSVM(ActiveModel):
    model: SVC

    def __init__(self):
        self.model = SVC(
            C=1,
            kernel="rbf",
            degree=3, # Only for `poly` kernel
        )

    def fit(self, x_init: X, y_init: Y, x: X, y: Y):
        self.model.fit(x_init, y_init)
        # TODO(marco): Continue
        print(self.model.support_vectors_)

    def predict(self, x: X) -> NDArray:
        return self.model.predict(x)


