import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        M=to_reshape.shape[0]
        N=to_reshape.shape[1]
        return torch.reshape(to_reshape,(M*N//2,2))
        # Use torch.reshape(tensor, new_shape)
        pass

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        return torch.mean(to_avg.float(),dim=0)
        pass

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=
        return torch.cat((cat_one,cat_two),dim=1)
        pass

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        return torch.nn.functional.mse_loss(prediction,target)
        pass
