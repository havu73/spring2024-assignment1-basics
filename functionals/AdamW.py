import torch
from collections.abc import Iterable, Callable
from typing import Optional
import math

'''
Author: Ha Vu with the help of Github Copilot
'''
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        '''
        :param params: iterable of parameters to optimize or dicts defining parameter groups
        :param lr: learning rate
        :param betas: coefficients used for computing running averages of gradient and its square
        :param eps: term added to the denominator to improve numerical stability
        :param weight_decay: weight decay (L2 penalty) (default: 0.01)
        '''
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self, closure:Optional[Callable]=None):
        loss = None if closure is None else closure()  # reevaluate the loss if closure is provided
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['t'] = 1  # t starts from 1 instead of 0
                    state['m'] = torch.zeros_like(p.data) # first moment estimate should be initialized to 0
                    state['v'] = torch.zeros_like(p.data)  # second moment estimate should be initialized to 0
                m, v, t = state['m'], state['v'], state['t']
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']
                lambd = group['weight_decay']
                state['m'] = beta1*m+(1-beta1)*grad
                state['v'] = beta2*v+(1-beta2)*grad**2
                alpha_t = (lr* math.sqrt(1-beta2**(t)) /
                           (1-beta1**(t)))
                p.data -= alpha_t * (state['m'] / (torch.sqrt(state['v']) + eps))
                p.data -= lambd * lr * p.data  # apply weight decay, to make the parameters get closer to zeros I believe
                state['t'] += 1
        return loss

