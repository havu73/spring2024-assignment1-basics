import torch
import numpy as np
import helper
import argparse
from Transformer.Transformer import TransformerLM
from dataloader.data_loader import get_batch
from functionals.loss import cross_entropy
from functionals.AdamW import AdamW
from functionals.learning_rate import lr_cosine_schedule
from checkpointing.checkpoint import save_checkpoint, load_checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BaseModelTrainer:
    def __init__(self, vocab_size: int=10000,
    seq_len: int= 512,
    d_model: int= 512,
    num_layers: int=6,
    num_heads: int=8,
    d_ff: int = 2048,
    attn_pdrop: float= 0,
    residual_pdrop: float=0,
    train_fn: str=None,
    val_fn: str =None,
    batch_size: int = 1024,
    checkpoint_dir: str= './checkpoints', *args, **kwargs):
        """
        Things I need in a model trainer:
        - Model: TransformerLM
        - Dataloader: provided by the user
        - Loss function: Cross Entropy
        - Optimizer
        - Training loop

        """
        self.model = TransformerLM(vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop)
        import pdb; pdb.set_trace()
        self.scheduler = None
        self.train_fn = train_fn
        self.val_fn = val_fn
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        helper.make_dir(checkpoint_dir)

    @staticmethod
    def open_data_file(fn):
        """
        Open the data file and return the data as a numpy array.
        """
        return np.load(fn, mmap_mode='r')

    def solve(self, lr=1e-3,
              num_epochs=1000,
              weight_decay=0.01,
              betas=(0.9, 0.999),
              eps= 1e-8,
              max_learning_rate=1,
              min_learning_rate=0.1,
              warmup_iters=7,
              cosine_cycle_iters=21,
              *args, **kwargs):
        """
        Training loop implementation.
        """
        optimizer = AdamW(self.model.parameters(), lr=lr,
                         betas=betas,
                        weight_decay=weight_decay,
                        eps=eps)
        self.scheduler = lambda t: lr_cosine_schedule(t, alpha_max= max_learning_rate, alpha_min= min_learning_rate, Tw=warmup_iters, Tc=cosine_cycle_iters)
        self.model.to(device)
        train_data = self.open_data_file(self.train_fn)
        val_data = self.open_data_file(self.val_fn)
        for epoch in range(num_epochs):
            # Step 1: Set model to training mode
            self.model.train()
            # Step 2: Get the learning rate from your custom scheduler
            lr = self.scheduler(epoch)
            # Step 3: Update the learning rate for the optimizer (if your optimizer supports it)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Step 4: Zero the gradients from the previous iteration
            optimizer.zero_grad()  # zero_grad() was implemented becaue AdamW inherits from torch.optim.Optimizer
            # Step 5: Forward pass - compute the model output
            inputs, targets = get_batch(train_data, batch_size=self.batch_size, seq_len=self.model.seq_len, device=device)  # I dont necessarily need the torch.utils.data.DataLoader class for this
            # inputs: (bs, seq_len) [x1,x2,x3]
            # targets: (bs, seq_len) [x2,x3,x4]
            outputs = self.model(inputs)  # outputs: (bs, seq_len, vocab_size)
            # Step 6: Compute the loss using the criterion
            loss = cross_entropy(outputs, targets)  # loss: scalar. Averaged out over the batch and the sequence length (bc for each sample, there are seq_len predictions being made)
            # Step 7: Backward pass - compute gradients
            loss.backward()  # this function is handled implicitly by the autograd engine to register the gradients for the model parameters
            # Step 8: Optimizer step - update the weights using the computed gradients
            optimizer.step()  # this step function was implemented in the AdamW class code.
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                save_fn = f"{self.checkpoints_dir}/epoch_{epoch}.pt"
                save_checkpoint(self.model, optimizer, epoch, save_fn)
        return

def get_group_params(args, group_name):
    return {
        action.dest: getattr(args, action.dest, None)
        for action in group_name._group_actions
        if action.dest != 'help'
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script with argument groups")
    model_group = parser.add_argument_group('Transformer Parameters')
    model_group.add_argument('--vocab_size', type=int, default=10000, help='Size of the vocabulary')
    model_group.add_argument('--seq_len', type=int, default=512, help='Length of the context window')
    model_group.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    model_group.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
    model_group.add_argument('--num_heads', type=int, default=8, help='Number of heads in the model')
    model_group.add_argument('--d_ff', type=int, default=2048, help='Dimension of the feedforward network')
    model_group.add_argument('--attn_pdrop', type=float, default=0.1, help='Dropout probability for attention layers')
    model_group.add_argument('--residual_pdrop', type=float, default=0.1, help='Dropout probability for residual connections')

    lr_group = parser.add_argument_group('parameters used for the learning rate scheduler')
    lr_group.add_argument('--max_learning_rate', type=float, default=1, help='Maximum learning rate')
    lr_group.add_argument('--min_learning_rate', type=float, default=0.1, help='Minimum learning rate')
    lr_group.add_argument('--warmup_iters', type=int, default=7, help='Number of warm-up iterations')
    lr_group.add_argument('--cosine_cycle_iters', type=int, default=21, help='Number of cosine cycle iterations')
    lr_group.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    lr_group.add_argument('--num_epochs', type=int, default=1024, help='Number of epochs')

    adamw_group = parser.add_argument_group('AdamW Parameters')
    adamw_group.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    adamw_group.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for AdamW')
    adamw_group.add_argument('--eps', type=float, default=1e-8, help='Epsilon for AdamW')

    # Create the second argument group: data parameters
    data_group = parser.add_argument_group('Data Parameters')
    data_group.add_argument('--train_fn', type=str, required=True, help='Directory for the dataset')
    data_group.add_argument('--val_fn', type=str, required=True, help='Number of data loading workers')

    # Create the third argument group: experiment parameters
    experiment_group = parser.add_argument_group('Experiment Parameters')
    experiment_group.add_argument('--batch_size', type=int, default=1000, help = 'batch size for each epoch')
    experiment_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    experiment_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')

    # Parse the arguments
    args = parser.parse_args()
    # for each group of arguments, we create a dictionary with the key and values of the arguments
    model_params = get_group_params(args, model_group)
    lr_params = get_group_params(args, lr_group)
    adamw_params = get_group_params(args, adamw_group)
    data_params = get_group_params(args, data_group)
    experiment_params = get_group_params(args, experiment_group)
    all_params = vars(args)
    # we pass the dictionaries to the BaseModelTrainer class
    trainer = BaseModelTrainer(**all_params)
    print ('Done getting command line arguments')

