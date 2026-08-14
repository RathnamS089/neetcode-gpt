import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
       n=y_pred.shape[0]
       a=y_true*np.log(y_pred)
       b=(1-y_true)*(np.log(1-y_pred))
       loss=-1/n*np.sum(a+b)
       return np.round(loss,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n=y_pred.shape[0]
        a=np.sum(y_true*np.log(y_pred))
        loss=-1/n*(np.sum(a))
        return np.round(loss,4)
        pass
