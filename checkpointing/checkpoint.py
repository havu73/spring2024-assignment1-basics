import torch

def save_checkpoint(model, optimizer, epoch, fn):
    """
    Save a PyTorch model checkpoint.
    :param model: torch.nn.Module, model to save
    :param optimizer: torch.optim.Optimizer, optimizer to save
    :param epoch: int, epoch number
    :param fn: str, filename to save the checkpoint to
    """
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, fn)

def load_checkpoint(fn, model, optimizer):
    '''
    load a PyTorch model checkpoint
    :param fn: str, filename of the checkpoint
    :param model: torch.nn.Module, model to load the checkpoint into
    :param optimizer: torch.optim.Optimizer, optimizer to load the checkpoint into
    '''
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return epoch