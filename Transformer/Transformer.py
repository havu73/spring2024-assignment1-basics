import torch
import torch.nn as nn
from .activation import softmax
from .attention import MultiHead
from .FFN import FFN
from .RmsNorm import RMSNorm

'''
Model accounting:
- rmsnorm1: d_model parameter (g)
- rmsnorm2: d_model parameter (g)
- multiheadattention (no bias): 3 * d_model * d_model 
- FFN: 2 * d_model * d_ff
In total: summing all the parameters above * 4byte per parameter --> ~ 107 MB
(112,652,800 parameters in total, without the bias terms)

How many FLOP operations are needed for the forward pass of the model?
- rmsnorm1: 3* d_model: squaring a_i, adding a_i's, multiplying a_i * g_i
- rmsnorm2: 3* d_model
- multiheadattention: 3 * 2 * d_model*d_model*seq_len: for each x of seq_len multiply by WQ, WK, WV. Each matrix*matrix multiplication costs 2*d_model*d_model*seq_len opeations
then, Attention(Q,K,V): 2 * seq_len * seq_len * d_model (for QK^T)
softmax(QK^T): 2 * seq_len * seq_len: getting the denom and also dividing each entry of seq_len*seq_len matrix by the appropriate denom
then, QK^T * V: 2 * seq_len * seq_len * d_model (for QK^T * V)
Given the specification of GPT-2XL, this will need: 22,439,526,400 FLOPs for just the matrix multiplications FOR ONE LAYER
If seq_len becomes 16384: 1,969,645,158,400 ~ 2 Trillion FLOPs FOR ONE LAYER
'''


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
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.residual_dropout = nn.Dropout(residual_pdrop)

    def forward(self,x):
        '''
        :param x: torch.Tensor: input tensor (bs1,..., bsN, seq_len, d_model)
        :return: torch.Tensor: output tensor
        '''
        # y=x+Dropout(MultiHead(RMSNorm(x)))
        # we skip dropout for now (not implemented yet)
        x_hat = self.rmsnorm1(x)  #(bs1,..., bsN, seq_len, d_model)
        x_hat = self.mha(x_hat)  # (bs1,..., bsN, seq_len, d_model)
        x_hat = self.attn_dropout(x_hat)  # (bs1,..., bsN, seq_len, d_model)
        x_hat = x_hat + x
        # x_hat will be input to the next part of the block: y=y+Dropout(FFN(RMSNorm(y)))
        # we skip dropout for now (not implemented yet)
        y = self.rmsnorm2(x_hat)  # (bs1,..., bsN, seq_len, d_model)  --> normalize by the last dimension
        y = self.ffn(y)  # (bs1,..., bsN, seq_len, d_model)
        y = self.residual_dropout(y) # (bs1,..., bsN, seq_len, d_model)
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

class TransformerLM(nn.Module):
    def __init__(self, num_tokens, seq_len, d_model=1600, embed_pdrop=0, num_heads=25, d_ff=6400, attn_pdrop=0, residual_pdrop=0, num_layers=48):
        '''
        :param num_tokens: int: the number of tokens in the vocabulary
        :param d_model: int: the number of features in the input tensor
        :param num_heads: int: the number of heads
        :param d_ff: int: the number of features in the feed-forward layer
        :param attn_pdrop: float: the dropout probability for the attention layer
        :param residual_pdrop: float: the dropout probability for the residual connections
        :param num_layers: int: the number of layers
        '''
        super(TransformerLM, self).__init__()
        # 1. parameters and model components associated with the embedding of the tokens and positions
        self.num_tokens = num_tokens  # vocab size
        self.seq_len = seq_len  # input sequence length
        self.d_model = d_model  # each token will be represented by a vector of size d_model, each position within seq_len will also be represented by a vector of size d_model
        self.token_embed = nn.Embedding(num_tokens, d_model)  # nn.Embedding will be simply a matrix of size (num_tokens, d_model) --> each token is represented by a vector of size d_model, and this matrix is learnable
        self.position_embed = nn.Embedding(seq_len, d_model)  # nn.Embedding will be simply a matrix of size (seq_len, d_model) --> each position is represented by a vector of size d_model, and this matrix is learnable
        self.embed_dropout = nn.Dropout(embed_pdrop)
        # 2. Parameters and model components associated with ONE transformer block
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.num_layers = num_layers
        self.transformer_list = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(d_model, bias=False)  # LayerNorm will be a vector of size d_model, and parametersl beta and gamma will be learned
        self.fc = nn.Linear(d_model, num_tokens, bias=False)

    def forward(self, x):
        '''
        :param x: torch.Tensor: input tensor (bs, seq_len)
        '''
        # for each token in the input sequence, we will first embed the token and the position
        token_embed = self.token_embed(x)  # (bs1, bs2, ..., bsN, seq_len, d_model)
        raw_position = torch.arange(x.size(1), device=x.device)  # (seq_len,)
        raw_position = raw_position[None,:] # (1, seq_len)
        raw_position = raw_position.expand(x.size(0), -1)  # (bs, seq_len): repeat number of times equal to batch_size, keep the last
        position_embed = self.position_embed(raw_position)  # (bs, seq_len, d_model)
        x = token_embed + position_embed # (bs, seq_len, d_model)
        x = self.embed_dropout(x)
        for transformer in self.transformer_list:
            x = transformer(x) # (bs, seq_len, d_model)
        x = self.ln(x)
        x = self.fc(x)  # (bs, seq_len, num_tokens) --> for each token in the sequence, we will predict the probability of the next token
        return x

    def load_state_dict_one_trans_layer(self, weights, layer_idx):
        '''
        :param weights: dict: the weights of the model
        :param layer_idx: int: the index of the layer
        reformat the weights of the model for one transformer layer to be a format that is compatible with the format accepted by TransformerBlock.load_state_dict
        '''
        weights_to_transformer = {}
        weights_to_transformer['attn.q_proj.weight'] = weights[f'layers.{layer_idx}.attn.q_proj.weight']
        weights_to_transformer['attn.k_proj.weight'] = weights[f'layers.{layer_idx}.attn.k_proj.weight']
        weights_to_transformer['attn.v_proj.weight'] = weights[f'layers.{layer_idx}.attn.v_proj.weight']
        weights_to_transformer['attn.output_proj.weight'] = weights[f'layers.{layer_idx}.attn.output_proj.weight']
        weights_to_transformer['ffn.w1.weight'] = weights[f'layers.{layer_idx}.ffn.w1.weight']
        weights_to_transformer['ffn.w2.weight'] = weights[f'layers.{layer_idx}.ffn.w2.weight']
        weights_to_transformer['ln1.weight'] = weights[f'layers.{layer_idx}.ln1.weight']
        weights_to_transformer['ln2.weight'] = weights[f'layers.{layer_idx}.ln2.weight']
        return weights_to_transformer

    def load_state_dict(self, weights):
        '''
        weights: dict[str, torch.FloatTensor]
        State dict of our reference implementation. {num_layers} refers to an
        integer between `0` and `num_layers - 1` (the layer index).
        The keys of this dictionary are:
        - `token_embeddings.weight`
            Token embedding matrix. Shape is (vocab_size, d_model).
        - `position_embeddings.weight`
            Positional embedding matrix. Shape is (context_length, d_model).
        - `layers.{num_layers}.attn.q_proj.weight`
            The query projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.k_proj.weight`
            The key projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.v_proj.weight`
            The value projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_v),
            so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.output_proj.weight`
            Weight of the multi-head self-attention output projection
            Shape is ((d_model / num_heads) * num_heads, d_model).
        - `layers.{num_layers}.ln1.weight`
            Weights of affine transform for the first RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `layers.{num_layers}.ffn.w1.weight`
            Weight of the first linear transformation in the FFN.
            Shape is (d_ff, d_model).
        - `layers.{num_layers}.ffn.w2.weight`
            Weight of the second linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `layers.{num_layers}.ln2.weight`
            Weights of affine transform for the second RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `ln_final.weight`
            Weights of affine transform for RMSNorm applied to the output of the final transformer block.
            Shape is (d_model, ).
        - `lm_head.weight`
            Weights of the language model output embedding.
            Shape is (vocab_size, d_model).
        '''
        self.token_embed.weight.data = weights['token_embeddings.weight']
        self.position_embed.weight.data = weights['position_embeddings.weight']
        for i in range(self.num_layers):
            weights_to_transformer = self.load_state_dict_one_trans_layer(weights, i)
            self.transformer_list[i].load_state_dict(weights_to_transformer)
        self.ln.weight.data = weights['ln_final.weight']
        self.fc.weight.data = weights['lm_head.weight']

