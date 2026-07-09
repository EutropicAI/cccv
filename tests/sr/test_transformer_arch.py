from typing import Callable

import pytest
import torch
from torch import nn

from cccv.arch.sr.dat_arch import DAT, Spatial_Attention
from cccv.arch.sr.hat_arch import HAT, WindowAttention as HATWindowAttention
from cccv.arch.sr.swinir_arch import SwinIR, WindowAttention as SwinIRWindowAttention


def _relative_position_index(window_size: int) -> torch.Tensor:
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1)


def _assert_bf16_attention_forward(forward: Callable[[], torch.Tensor]) -> None:
    try:
        out = forward()
    except RuntimeError as e:
        message = str(e)
        lower_message = message.lower()
        if "bfloat16" in lower_message and ("not implemented" in lower_message or "unsupported" in lower_message):
            pytest.skip(f"bfloat16 attention is not supported by this torch build: {message}")
        raise

    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


def test_dat_attention_mask_matches_bf16_dtype() -> None:
    def forward() -> torch.Tensor:
        attn = Spatial_Attention(dim=8, idx=0, split_size=[4, 4], num_heads=2).to(torch.bfloat16).eval()
        qkv = torch.randn((3, 1, 16, 8), dtype=torch.bfloat16)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        with torch.inference_mode():
            return attn(qkv, 4, 4, mask=mask)

    _assert_bf16_attention_forward(forward)


def test_dat_relative_position_bias_matches_bf16_dtype() -> None:
    def forward() -> torch.Tensor:
        attn = Spatial_Attention(dim=8, idx=0, split_size=[4, 4], num_heads=2).eval()
        qkv = torch.randn((3, 1, 16, 8), dtype=torch.bfloat16)
        with torch.inference_mode():
            return attn(qkv, 4, 4)

    _assert_bf16_attention_forward(forward)


def test_hat_attention_mask_matches_bf16_dtype() -> None:
    def forward() -> torch.Tensor:
        attn = HATWindowAttention(dim=8, window_size=(4, 4), num_heads=2).to(torch.bfloat16).eval()
        x = torch.randn((1, 16, 8), dtype=torch.bfloat16)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        with torch.inference_mode():
            return attn(x, rpi=_relative_position_index(4), mask=mask)

    _assert_bf16_attention_forward(forward)


def test_swinir_attention_mask_matches_bf16_dtype() -> None:
    def forward() -> torch.Tensor:
        attn = SwinIRWindowAttention(dim=8, window_size=(4, 4), num_heads=2).to(torch.bfloat16).eval()
        x = torch.randn((1, 16, 8), dtype=torch.bfloat16)
        mask = torch.zeros((1, 16, 16), dtype=torch.float32)
        with torch.inference_mode():
            return attn(x, mask=mask)

    _assert_bf16_attention_forward(forward)


def _dat_model() -> nn.Module:
    return DAT(
        img_size=16,
        embed_dim=16,
        split_size=[4, 8],
        depth=[3],
        num_heads=[4],
        expansion_factor=2,
        drop_path_rate=0.0,
        scale=2,
        upsampler="pixelshuffledirect",
    ).eval()


def _hat_model() -> nn.Module:
    return HAT(
        img_size=16,
        embed_dim=16,
        depth=[2],
        num_heads=[4],
        window_size=4,
        compress_ratio=4,
        squeeze_factor=4,
        mlp_ratio=2,
        drop_path_rate=0.0,
        scale=2,
        upsampler="pixelshuffle",
    ).eval()


def _swinir_model() -> nn.Module:
    return SwinIR(
        img_size=16,
        embed_dim=16,
        depths=[2],
        num_heads=[4],
        window_size=4,
        mlp_ratio=2,
        drop_path_rate=0.0,
        scale=2,
        upsampler="pixelshuffledirect",
    ).eval()


@pytest.mark.parametrize("make_model", [_dat_model, _hat_model, _swinir_model])
def test_transformer_arch_forward_shape(make_model: Callable[[], nn.Module]) -> None:
    model = make_model()
    x = torch.rand((1, 3, 16, 16))

    with torch.inference_mode():
        out = model(x)

    assert out.shape == (1, 3, 32, 32)
    assert out.dtype == torch.float32
