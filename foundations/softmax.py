import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        arr=np.empty(z.shape)
        max=z.max()
        sum=np.sum(np.exp(z-max))
        for i in range(len(z)):
           arr[i]=np.exp(z[i]-max)/sum
        return np.round(arr,4)
        pass
