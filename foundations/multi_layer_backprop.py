import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        n=y_true.shape[0]
        z1_before=x@W1.T+b1
        z1=np.maximum(z1_before,0)
        z2=z1@W2.T+b2
        loss = 1/n * np.sum((z2 - y_true)**2)
        dloss_dz2=2/n*(z2-y_true)
        dloss_dz1=dloss_dz2@W2
        dloss_dz1_before=dloss_dz1*(z1_before>0)
        dloss_dw2 = np.outer(dloss_dz2, z1)
        dloss_db2=dloss_dz2*1
        dloss_dw1 = np.outer(dloss_dz1_before, x)
        dloss_db1=dloss_dz1_before*1
        loss=np.round(loss,4)
        dloss_dw1=np.round(dloss_dw1,4)
        dloss_db1=np.round(dloss_db1,4)
        dloss_dw2=np.round(dloss_dw2,4)
        dloss_db2=np.round(dloss_db2,4)
        return {
            'loss':loss,
             'dW1':dloss_dw1,
             'db1':dloss_db1,
             'dW2':dloss_dw2,
             'db2':dloss_db2
        }
        pass
