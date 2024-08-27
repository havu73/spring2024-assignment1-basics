import torch

def cross_entropy_ind_points(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    :param inputs: torch.Tensor: the input tensor of shape (bs1, ..., bsN, d)
    :param targets: torch.Tensor: the target tensor of shape (bs1,...,bsN) --> index of the correct class
    :return: torch.Tensor: the cross-entropy loss, implemented from scratch
    """
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    log_sum_exp = inputs.exp().sum(dim=-1).log()  # sum_{i=1}^{D} exp(x_i) --> (bs1, ..., bsN)
    grid_indices = [torch.arange(size) for size in targets.shape]  # [0,..,bs1-1], ..., [0,..,bsN-1]
    grid_indices = torch.meshgrid(*grid_indices, indexing='ij')  # meshgrid of indices
    p_target = inputs[(*grid_indices, targets)]  # p_target = x_{target} for each point in targets (bs1, ..., bsN)
    return log_sum_exp - p_target  # (bs1, ..., bsN)


def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
    '''
    :param inputs: torch.Tensor: the input tensor of shape (bs1, ..., bsN, d)
    :param targets: torch.Tensor: the target tensor of shape (bs1,...,bsN) --> index of the correct class
    :return: torch.Tensor: the cross-entropy loss, implemented from scratch, average across all datapoints
    '''
    return cross_entropy_ind_points(inputs, targets).mean()