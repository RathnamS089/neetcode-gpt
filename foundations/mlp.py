import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        x=np.array(x)
        for l in range(len(weights)):
            w=np.array(weights[l])
            b=np.array(biases[l])
            z=np.dot(x,w)+b
            x=np.maximum(z,0)
        return np.round(x,5)
        pass
