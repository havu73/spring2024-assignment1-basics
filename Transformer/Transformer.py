import torch
import torch.nn as nn
from .activation import softmax
from .attention import MultiHead
from .FFN import FFN
from .RmsNorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop):
        '''
        :param d_model: int: the number of features in the input tensor
        :param num_heads: int: the number of heads
        :param d_ff: int: the number of features in the feed-forward layer
        :param attn_pdrop: float: the dropout probability for the attention layer
        :param residual_pdrop: float: the dropout probability for the residual connections
        :param num_layers: int: the number of layers
        '''
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.d_v = d_model//num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        # below are all the submodules of the transformer block
        self.ffn = FFN(d_model=self.d_model, d_ff=self.d_ff)
        self.mha = MultiHead(d_model=self.d_model, num_heads=self.num_heads, d_k=self.d_model//self.num_heads, d_v=self.d_model//self.num_heads)
        self.rmsnorm1 = RMSNorm(d=self.d_model)
        self.rmsnorm2 = RMSNorm(d=self.d_model)

    def forward(self,x):
        '''
        :param x: torch.Tensor: input tensor (bs1,..., bsN, seq_len, d_model)
        :return: torch.Tensor: output tensor
        '''
        # y=x+Dropout(MultiHead(RMSNorm(x)))
        # we skip dropout for now (not implemented yet)
        x_hat = self.rmsnorm1(x)
        x_hat = self.mha(x_hat)
        x_hat = x_hat + x
        # x_hat will be input to the next part of the block: y=y+Dropout(FFN(RMSNorm(y)))
        # we skip dropout for now (not implemented yet)
        y = self.rmsnorm2(x_hat)
        y = self.ffn(y)
        y = y + x_hat
        return y


    def load_state_dict(self, weights):
        '''
        :param weights: dict: the weights of the model
        '''
        attn_weights = {}
        for i in range(self.num_heads):
            attn_weights[f'q_heads.{i}.weight'] = weights[f'attn.q_proj.weight'][self.d_k*i:self.d_k*(i+1),:] # extract out a matrix of size (d_k, d_model)
            attn_weights[f'k_heads.{i}.weight'] = weights[f'attn.k_proj.weight'][self.d_k*i:self.d_k*(i+1),:]
            attn_weights[f'v_heads.{i}.weight'] = weights[f'attn.v_proj.weight'][self.d_k*i:self.d_k*(i+1),:]
        attn_weights['output_proj.weight'] = weights['attn.output_proj.weight']  # extract out a matrix of size (d_v * num_heads, d_model)
        ffn_weights = {}
        ffn_weights['w1.weight'] = weights['ffn.w1.weight']
        ffn_weights['w2.weight'] = weights['ffn.w2.weight']
        rmsnorm1_weights = {}
        rmsnorm1_weights['g'] = weights['ln1.weight']  # extract out a vector of size (d_model)
        rmsnorm2_weights = {}
        rmsnorm2_weights['g'] = weights['ln2.weight']
        self.mha.load_state_dict(attn_weights)
        self.ffn.load_state_dict(ffn_weights)
        self.rmsnorm1.load_fixed_gain(rmsnorm1_weights)
        self.rmsnorm2.load_fixed_gain(rmsnorm2_weights)
        return

