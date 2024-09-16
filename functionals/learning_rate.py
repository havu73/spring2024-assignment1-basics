import torch

def lr_cosine_schedule(t, alpha_max= 1, alpha_min= 0.1, Tw=7, Tc=21):  # default params values based on test_optimizer.py
    '''
    :param t: int: the current iteration
    :param alpha_max: float: the maximum learning rate
    :param alpha_min: float: the minimum learning rate
    :param Tw: int: the number of warm-up iterations
    :param Tc: int: the index of the end of cosine cycle iterations
    :return: float: the learning rate at iteration t
    '''
    if t < Tw:
        return alpha_max * t / Tw
    elif t <= Tc:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + torch.cos(torch.tensor(t - Tw) * 3.141592653589793 / (Tc-Tw)))
    else:  # t > Tc
        return alpha_min