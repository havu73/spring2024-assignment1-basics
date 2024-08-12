#!/usr/bin/env python3
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_gelu,
    run_multihead_self_attention,
    run_positionwise_feedforward,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
)
from .common import FIXTURES_PATH


def test_positionwise_feedforward():
    reference_weights = torch.load(
        FIXTURES_PATH / "positionwise_feedforward_weights.pt"
    )
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(
        FIXTURES_PATH / "positionwise_feedforward_expected_output.pt"
    )
    d_model = 64
    d_ff = 128

    actual_output = run_positionwise_feedforward(
        d_model=d_model, d_ff=d_ff, weights=reference_weights, in_features=in_features
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_scaled_dot_product_attention():
    torch.manual_seed(42)
    # Take the first batch item, so we test the 3D case
    # (input shape (batch_size, seq_len, d_k)) for scaled dot-product attention.
    K = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_K.pt")[0]
    Q = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_Q.pt")[0]
    V = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_V.pt")[0]
    mask = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_mask.pt")
    pdrop = 0.0
    expected_output = torch.load(
        FIXTURES_PATH / "scaled_dot_product_attention_expected_output.pt"
    )[0]
    actual_output = run_scaled_dot_product_attention(
        K=K, Q=Q, V=V, mask=mask, pdrop=pdrop
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_4d_scaled_dot_product_attention():
    torch.manual_seed(42)
    # Shape: (batch_size, num_heads, seq_len, d_k)
    K = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_K.pt")
    Q = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_Q.pt")
    V = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_V.pt")
    mask = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_mask.pt")
    pdrop = 0.0
    expected_output = torch.load(
        FIXTURES_PATH / "scaled_dot_product_attention_expected_output.pt"
    )
    actual_output = run_scaled_dot_product_attention(
        K=K, Q=Q, V=V, mask=mask, pdrop=pdrop
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_multihead_self_attention():
    reference_weights = torch.load(
        FIXTURES_PATH / "unbatched_multihead_self_attention_weights.pt"
    )
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(
        FIXTURES_PATH / "unbatched_multihead_self_attention_expected_output.pt"
    )
    d_model = 64
    num_heads = 2
    d_k = d_model // num_heads
    attn_pdrop = 0.0
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        attn_pdrop=attn_pdrop,
        weights=reference_weights,
        in_features=in_features,
    )
    # get the output from torch
    multihead_attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=False)
    new_in_proj_weight = torch.ones((3 * d_model, d_model))  # (3 * d_model) x d_model
    for i in range(num_heads):
        new_in_proj_weight[i * d_k : (i + 1) * d_k] = reference_weights[f"q_heads.{i}.weight"]
        new_in_proj_weight[(num_heads + i) * d_k : (num_heads + i + 1) * d_k] = reference_weights[f"k_heads.{i}.weight"]
        new_in_proj_weight[(2 * num_heads + i) * d_k : (2 * num_heads + i + 1) * d_k] = reference_weights[f"v_heads.{i}.weight"]
    new_out_proj_weight = reference_weights["output_proj.weight"]
    # now assign the new weights to the multi-head attention layer
    multihead_attn.in_proj_weight.data = new_in_proj_weight
    multihead_attn.out_proj.weight.data = new_out_proj_weight
    # get the output from the multi-head attention layer
    expected_output, _ = multihead_attn(in_features, in_features, in_features)
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )

def test_multihead_self_attention_ha():
    torch.manual_seed(42)
    embed_dim = 64
    num_heads = 2
    d_k = embed_dim // num_heads
    # Create the MultiheadAttention layer
    multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)
    # Define new weights for the query, key, value projections and output projection
    new_in_proj_weight = torch.ones((3 * embed_dim, embed_dim))  # (3 * embed_dim) x embed_dim
    new_in_proj_weight = torch.rand(new_in_proj_weight.shape).float()  # random values
    # new_in_proj_weight[0*num_heads*d_k:1*(num_heads*d_k)] = torch.randint(0,10, new_in_proj_weight[0*num_heads*d_k:1*num_heads*d_k].shape)  # random values
    new_out_proj_weight = torch.rand((embed_dim, embed_dim))  # embed_dim x embed_dim
    # new_out_proj_weight[:,1] = new_out_proj_weight[:,1] *2
    torch.set_printoptions(precision=3)
    # Assign new weights to the multi-head attention layer
    multihead_attn.in_proj_weight.data = new_in_proj_weight
    multihead_attn.out_proj.weight.data = new_out_proj_weight

    in_features = torch.rand((8, 128, embed_dim))  # (batch_size, seq_len, embed_dim)
    out_features, attention_weights = multihead_attn(in_features, in_features, in_features)
    reference_weights = {}
    for i in range(num_heads):
        reference_weights[f"q_heads.{i}.weight"] = new_in_proj_weight[i * d_k : (i + 1) * d_k]
        reference_weights[f"k_heads.{i}.weight"] = new_in_proj_weight[(num_heads + i) * d_k : (num_heads + i + 1) * d_k]
        reference_weights[f"v_heads.{i}.weight"] = new_in_proj_weight[(2 * num_heads + i) * d_k : (2 * num_heads + i + 1) * d_k]
    reference_weights["output_proj.weight"] = new_out_proj_weight
    actual_output = run_multihead_self_attention(
        d_model=embed_dim,
        num_heads=num_heads,
        attn_pdrop=0,
        weights=reference_weights,
        in_features=in_features,
    )
    numpy.testing.assert_allclose(out_features.detach().numpy(), actual_output.detach().numpy(), atol=1e-6, rtol=1e-06)

def test_transformer_lm():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices = torch.load(FIXTURES_PATH / "in_indices.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_lm_expected_output.pt")
    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-4
    )


def test_transformer_lm_truncated_input():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices_truncated = torch.load(FIXTURES_PATH / "in_indices_truncated.pt")
    truncated_expected_output = torch.load(
        FIXTURES_PATH / "transformer_lm_truncated_expected_output.pt"
    )
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices_truncated,
    )
    numpy.testing.assert_allclose(
        truncated_actual_output.detach().numpy(),
        truncated_expected_output.detach().numpy(),
        atol=1e-4,
    )


def test_transformer_block():
    torch.manual_seed(42)
    reference_weights = torch.load(FIXTURES_PATH / "transformer_block_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_block_expected_output.pt")
    d_model = 64
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_features=in_features,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_rmsnorm():
    reference_weights = torch.load(FIXTURES_PATH / "rmsnorm_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "rmsnorm_expected_output.pt")
    d_model = 64
    actual_output = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_features
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_gelu():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = torch.tensor(
        [
            [
                0.13946731388568878,
                0.7617851495742798,
                0.3622361421585083,
                0.3221103549003601,
                0.8121858239173889,
            ],
            [
                0.5881373286247253,
                0.8080969452857971,
                0.1243969276547432,
                0.7709409594535828,
                0.007538566831499338,
            ],
        ]
    )
    actual_output = run_gelu(x)
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_gelu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.gelu(x)
    actual_output = run_gelu(x)
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )
