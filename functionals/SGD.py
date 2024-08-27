from collections.abc import Iterable, Callable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        '''
        :param params: iterable of parameters to optimize or dicts defining parameter groups
        :param lr: learning rate
        '''
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {'lr':lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()  # reevaluate the loss if closure is provided
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] # get the state associated with p
                t=state.get('t', 0) # number of iterations so far,or 0 if None
                d_p = p.grad.data
                p.data -= lr /math.sqrt(t+1)* d_p
                self.state[p]['t'] = t+1
        return loss

if __name__ == '__main__':
    '''
    Here I just use some code to play with the optimizer.
    Basically, the optimizer is supposed to minimize the loss function, which is the sum of the squares of the weights.
    So, optimizer should reduce the weights to as close to zeros as possible.
    We can run this code with multiple learning rates to see how the weights are calculated at the end of training with the same number of epochs.
    '''
    weights = torch.nn.Parameter(5*torch.ones(10,10))
    opt = SGD([weights], lr=0.003)
    for i in range(100):
        opt.zero_grad()
        loss = (weights**2).sum()
        loss.backward()
        opt.step()
        print(loss.item())
    print(weights)