import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        intial=np.dot(w,x)
        z=np.sum(intial)+b
        if activation=="sigmoid":
            act=1/(1+np.exp(-z))
        if activation=="relu":
            act=max(0,z)
        return float(np.round(act,5))
        pass
