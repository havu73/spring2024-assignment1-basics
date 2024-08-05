'''
This file will implmeent the positionwise feedforward network linear transformation that is commonly used in the Transformer model
FFN(x) = GELU(xW1)W2.
We droped the bias terms here (if bias is present, FFN(X) = GELU(XW1 + b1)W2 + b2)
W1: (d_model, d_ff)
W2: (d_ff, d_model)
d_ff: the number of neurons in the hidden layer, usually 4*d_model
Author: Ha Vu with the help of Github Copilot
'''

import torch
import torch.nn as nn
from .GELU import GELU

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        '''
        :param d_model: int: the number of features in the input tensor
        :param d_ff: int: the number of neurons in the hidden layer
        '''
        super(FFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = GELU

    def forward(self, x: torch.Tensor, W1: torch.Tensor=None, W2: torch.Tensor=None) -> torch.Tensor:
        '''
        :param x: torch.Tensor: input tensor
        :return: torch.Tensor: output tensor
        '''
        if W1 is not None:
            self.w1.weight = W1
        if W2 is not None:
            self.w2.weight = W2
        return self.w2(self.activation(self.w1(x)))

    def load_state_dict(self, state_dict: dict):
        '''
        :param state_dict: dict: the state dictionary
        '''
        self.w1.weight.data = state_dict["w1.weight"]
        self.w2.weight.data = state_dict["w2.weight"]
        return self