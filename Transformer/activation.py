'''
This file should include a few activation functions that are commonly used in deep learning models
Author: Ha Vu with the help of Github Copilot
'''

import torch
def softmax(x, dim:int=None):
    '''
    :param x: torch.Tensor: input tensor
    :param i: int: the dimension to apply softmax, if Nont apply softmax to the last dimension
    :return: torch.Tensor: output tensor
    '''
    if dim is None:
        dim = x.dim() - 1
    # first, for each row, subtract the maximum value of that row
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x = x - max_x
    # now apply the softmax
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

