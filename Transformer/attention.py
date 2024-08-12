'''
This file will implement the scaled dot-product attention mechanism that is commonly used in the Transformer model
Author: Ha Vu with the help of Github Copilot
'''
import torch
import torch.nn as nn
from .activation import softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, d_v: int):
        '''
        :param d_k: int: the dimension of the key and query vectors
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        '''
        :param Q: torch.Tensor: query tensor (bs1, ..., bsN, seq_lenQ, d_k)
        :param K: torch.Tensor: key tensor (bs1,..., bsN, seq_lenK, d_k)
        :param V: torch.Tensor: value tensor (bs1,..., bsN, seq_lenK, d_v)
        :param mask: torch.Tensor: mask tensor (seq_lenQ, seq_lenK)
        :return: torch.Tensor: output tensor
        '''
        assert Q.size(-1) == self.d_k
        assert K.size(-1) == self.d_k
        assert V.size(-1) == self.d_v
        assert mask is None or mask.size() == (Q.size(-2), K.size(-2))
        K_transposed = K.transpose(-2, -1)  # (bs1,..., bsN, d_k, seq_lenK)
        # compute the scaled dot-product attention
        scores = torch.matmul(Q, K_transposed) / torch.sqrt(torch.tensor(self.d_k).float()) # (bs1,..., bsN, seq_lenQ, seq_lenK)
        if mask is not None:
            mask = mask.to(scores.dtype) * -1e9 # (seq_lenQ, seq_lenK)
            scores = scores + mask  #(bs1,..., bsN, seq_lenQ, seq_lenK)
        attention = softmax(scores, dim=-1) #(bs1,..., bsN, seq_lenQ, seq_lenK) such that the last dimension sums to 1
        return torch.matmul(attention, V)

class MultiHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int):
        '''
        :param d_model: int: the number of features in the input tensor
        :param num_heads: int: the number of heads
        :param d_k: int: the dimension of the key and query vectors
        :param d_v: int: the dimension of the value vectors
        '''
        super(MultiHead, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.WQ = nn.Linear(d_model, d_k*num_heads, bias=False)  # weight matrix will be (output, input) in nn.Linear
        self.WK = nn.Linear(d_model, d_k*num_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v*num_heads, bias=False)
        self.WO = nn.Linear(d_v*num_heads, d_model, bias=False)
        self.attention = ScaledDotProductAttention(d_k, d_v)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        '''
        :param x: torch.Tensor: input tensor (bs1,..., bsN, seq_len, d_model)
        :param mask: torch.Tensor: mask tensor (seq_len, seq_len)
        :return: torch.Tensor: output tensor
        '''
        bs = x.size()[:-2]
        seq_len = x.size(-2)
        Q = self.WQ(x).view(*bs, seq_len, self.num_heads, self.d_k).transpose(-3, -2)  # (bs, num_heads, seq_len, d_k)
        K = self.WK(x).view(*bs, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = self.WV(x).view(*bs, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
        output = self.attention(Q, K, V, mask).transpose(-3, -2).contiguous()  # bs, seq_len, num_heads, d_v)
        output = output.view(*bs, seq_len, self.d_v*self.num_heads)  #(bs, seq_len, d_v*num_heads)
        return self.WO(output)

    def load_state_dict(self, state_dict: dict):
        '''
        :param state_dict: dict: the state dictionary
        '''
        # WQ should be concatnative of all state_dict['q_heads.i.weight'] for i in range(num_heads)
        # each state_dict['q_heads.i.weight'] has shape (d_k, d_model)
        WQ = torch.cat([state_dict[f"q_heads.{i}.weight"] for i in range(self.num_heads)], dim=0)  # (d_k*num_head, d_model)
        self.WQ.weight.data = WQ  # (d_k*num_head, d_model) which is (d_output, d_input) in nn.Linear
        # WK should be concatnative of all state_dict['k_heads.i.weight'] for i in range(num_heads)
        # each state_dict['k_heads.i.weight'] has shape (d_k, d_model)
        WK = torch.cat([state_dict[f"k_heads.{i}.weight"] for i in range(self.num_heads)], dim=0)
        self.WK.weight.data = WK
        # WV should be concatnative of all state_dict['v_heads.i.weight'] for i in range(num_heads)
        # each state_dict['v_heads.i.weight'] has shape (d_v, d_model)
        WV = torch.cat([state_dict[f"v_heads.{i}.weight"] for i in range(self.num_heads)], dim=0)
        self.WV.weight.data = WV
        # WO from state_dict['o.weight'] (d_value * num_heads, d_model)
        W0 = state_dict["output_proj.weight"]
        self.WO.weight.data = W0  # (d_model, d_value * num_heads)
        # import pdb; pdb.set_trace()
        return self