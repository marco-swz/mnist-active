from typing import Annotated, Literal, Self, Optional
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray, ArrayLike
import keras
from statsmodels.stats.proportion import proportion_confint
import skopt
import pandas as pd
from datetime import datetime
import os

X = Annotated[NDArray[np.float64], Literal[28, 28]]
Y = np.int32

MAX_NUM_LABELED = 500
NUM_TEST = 100

class DataPoint:
    _num_labeled = 0
    _y: Y
    x: X
    is_labeled: bool = False

    def __init__(self, x, y):
        self.x = x
        self._y = y

    def get_label(self) -> np.int32:
        if not self.is_labeled:
            DataPoint._num_labeled += 1
            if DataPoint._num_labeled > MAX_NUM_LABELED:
                raise PermissionError("No more labels allowed!")

        self.is_labeled = True
        return self._y

class ActiveModel(ABC):
    @abstractmethod
    def fit(self, data_unlabeled: ArrayLike):
        ...

    @abstractmethod
    def predict(self, x: X) -> NDArray:
        ...

    @abstractmethod
    def get_params(self, deep: bool=True) -> dict:
        ...

    @abstractmethod
    def set_params(self, **params) -> Self:
        ...

def load_data() -> NDArray:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    xs = np.concatenate([x_train, x_test])
    ys = np.concatenate([y_train, y_test])
    data = np.array([DataPoint(x, y) for x, y in zip(xs, ys)])
    return data

def split_data(data: NDArray, test_ratio: float) -> tuple[NDArray, NDArray]:
    idxs_all = np.random.permutation(np.arange(len(data)))
    idx_split = int(test_ratio*MAX_NUM_LABELED)
    idxs_test = idxs_all[:idx_split]
    idxs_train = idxs_all[idx_split:]

    data_test = data[idxs_test]
    data_train = data[idxs_train]

    return data_train, data_test

def eval_model(model: ActiveModel, x, y) -> tuple[float, tuple[float, float]]:
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

def optimizer_score(self, data_full):
    data = list(filter(lambda d: not d.is_labeled, data_full))
    data = np.random.choice(data, MAX_NUM_LABELED - DataPoint._num_labeled, replace=False)
    x = np.array([d.x for d in data])
    y = np.array([d.get_label() for d in data])
    predictions = self.predict(x)
    num_correct = sum(predictions == y)
    accuracy = num_correct/len(predictions)

    DataPoint._num_labeled = 0
    for d in data_full:
        d.is_labeled = False
    return accuracy

def optimize_model(model: ActiveModel, opt_params: dict, data: NDArray):
    '''Finds and return the best classifier parameters given a provided search space.'''
    
    optimizer = skopt.BayesSearchCV(
        estimator=model, 
        search_spaces=opt_params,
        n_iter=1, 
        cv=3,
        refit=True, 
        n_jobs=1,
        scoring=optimizer_score,
        verbose=2,
        error_score=float("inf"), # pyright: ignore
    )

    optimizer.fit(data)

    result_table = pd.DataFrame(optimizer.cv_results_)\
        .sort_values('rank_test_score')\
        .set_index('rank_test_score')[['params', 'mean_test_score', 'std_test_score']]

    print(result_table)

    outdir = './data_opt'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    file_name = f'data_opt/opt_params_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'

    result_table.to_csv(file_name)

    return optimizer.best_params_

if __name__ == "__main__":
    pass
    
