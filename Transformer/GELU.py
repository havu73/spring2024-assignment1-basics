'''
This class will implmeent the GELU activation function
GELU(x) = 0.5 * x * (1 + P(Z <= x)) where Z ~ N(0,1)
GELU(x) = 0.5 * x * (1 + ef(x/sqrt(2)) where ef is the error function
The function GELUC can be approximated as:
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
where x is the input tensor
Usually, the pytorch implementation of GELU is used for better performance, but here I am learning to do it from scratch
Author: Ha Vu with the help of Github Copilot
'''

import torch
import math

def GELU(x: torch.Tensor) -> torch.Tensor:
    '''
    :param x: torch.Tensor: input tensor
    :return: torch.Tensor: output tensor
    '''
    return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))