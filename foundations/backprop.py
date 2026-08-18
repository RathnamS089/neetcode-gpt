import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        initial=np.dot(x,w)
        z=initial+b
        out=1/(1+np.exp(-z))
        error=1/2*((out-y_true)**2)
        dellwrtw=(out-y_true)*(out*(1-out))*x
        dellwrtb=(out-y_true)*(out*(1-out))
        dellwrtw=np.round(dellwrtw,5)
        dellwrtb=np.round(dellwrtb,5)
        return (dellwrtw,dellwrtb)
        pass
