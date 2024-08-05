'''
This class will do the root mean square normalization of the input tensor
a \in R^{d}: input tensor
RMSNorm(a_i) = a_i / RMS(a) * g_i
where:
- RMS(a) = sqrt(1/d * \sum_{i=1}^{d} a_i^2 + eps)
- g_i is a learnable parameter, called the gain

Author: Ha Vu and Github Copilot
'''

import torch
from torch import nn
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        '''
        :param d: int: the dimension of the input tensor
        :param eps: float: epsilon to prevent division by zero
        '''
        super(RMSNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, a: torch.Tensor, weights: torch.Tensor=None) -> torch.Tensor:
        '''
        :param a: torch.Tensor: input tensor of shape (n, d)
        :param weights: torch.Tensor: weights of shape (n, d), optional.
        If provided, the output will be a * weights / RMS(a). so weights here would replace g
        :return: torch.Tensor: normalized tensor of shape (n, d)
        '''
        # calculate the root mean square
        rms = torch.sqrt(torch.mean(a ** 2, dim=-1) + self.eps)
        if weights is not None:
            return a/ rms.unsqueeze(-1) * weights
        else:
            return a/ rms.unsqueeze(-1) * self.g
