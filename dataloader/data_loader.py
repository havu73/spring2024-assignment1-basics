import torch
import numpy as np
import numpy.typing as npt
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset: np.array
            1D numpy array of integer token IDs in the dataset.
        batch_size: int
            Desired batch size to sample.
        context_length: int
            Desired context length of each sampled example.
        device: str
            PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # randomly sample starting indices for each batch
    starting_indices = np.random.choice(len(dataset) - context_length, size=batch_size, replace=False)
    # extract the context windows from the dataset
    context_windows = np.array(
        [dataset[idx : idx + context_length] for idx in starting_indices]
    )
    # extract the target labels from the dataset
    labels = np.array(
        [dataset[idx + 1 : idx + context_length + 1] for idx in starting_indices]
    )
    # convert the context windows and labels to PyTorch tensors
    context_windows = torch.LongTensor(context_windows).to(device)
    labels = torch.LongTensor(labels).to(device)
    return context_windows, labels



from torch.utils.data import Dataset, DataLoader
class TokenizedDataset(Dataset):
    def __init__(self, file_path, seq_length=3):
        # Load tokenized data lazily using memory-mapping
        self.data = np.load(file_path, mmap_mode='r')
        self.seq_length = seq_length

    def __len__(self):
        # The number of sequences we can extract is len(data) - seq_length
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence: [x1, x2, x3]
        input_seq = self.data[idx:idx + self.seq_length]
        # Target sequence: [x2, x3, x4]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

