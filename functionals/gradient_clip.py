import torch

def gradient_clip(params, max_norm, eps=1e-6):
    '''
    :param params: torch.Tensor: the parameters of the model
    :param max_norm: float: the maximum norm of the gradients
    :return: None
    '''
    concat_params = torch.cat([p.grad.view(-1) for p in params if p.grad is not None])
    norm = torch.norm(concat_params, 2)
    # For gradient clipping, we need to do global gradient norm clipping (concat all the gradient, calculate the norm and scale the gradient by that norm)
    if norm > max_norm:
        for p in params:
            if p.grad is not None:
                p.grad *= max_norm / (norm + eps)
    return

