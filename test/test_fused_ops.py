import pytest
import torch

from vggt.graph import ACLGraphBlockRunner, GraphConfig
from vggt.layers.attention import Attention, Attention_fused
from vggt.layers.block import Block
from vggt.layers.mlp import Mlp, Mlp_fused
from vggt.layers.rope import RotaryPositionEmbedding2D

def _require_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pytest.skip("torch_npu is not available")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")

    return torch.device("npu")


def _build_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(seq_len, device=device, dtype=torch.int64)
    y = token_ids // 31
    x = token_ids % 31
    pos = torch.stack((y, x), dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1).clone()


def _build_attention_pair(device: torch.device):
    torch.manual_seed(42)

    ref_attn = Attention(
        dim=1024,
        num_heads=16,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=True,
        fused_attn=True,
        rope=RotaryPositionEmbedding2D(frequency=100),
    ).to(device=device, dtype=torch.float32)

    fused_attn = Attention_fused(
        dim=1024,
        num_heads=16,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=True,
        fused_attn=True,
        rope=RotaryPositionEmbedding2D(frequency=100),
    ).to(device=device, dtype=torch.float32)

    fused_attn.load_state_dict(ref_attn.state_dict(), strict=True)
    ref_attn.eval()
    fused_attn.eval()
    return ref_attn, fused_attn


def test_attention_fused_state_dict_compatible():
    device = _require_npu()
    ref_attn, fused_attn = _build_attention_pair(device)

    ref_keys = set(ref_attn.state_dict().keys())
    fused_keys = set(fused_attn.state_dict().keys())
    assert ref_keys == fused_keys


def test_attention_fused_forward_matches_reference():
    device = _require_npu()
    ref_attn, fused_attn = _build_attention_pair(device)

    torch.manual_seed(42)
    x = torch.randn(5, 930, 1024, device=device, dtype=torch.float32)
    pos = _build_positions(batch_size=5, seq_len=930, device=device)

    with torch.no_grad():
        ref_output = ref_attn(x, pos=pos)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        fused_output = fused_attn(x, pos=pos)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()

    ref_output = torch.nan_to_num(ref_output, nan=0.0, posinf=0.0, neginf=0.0)
    fused_output = torch.nan_to_num(fused_output, nan=0.0, posinf=0.0, neginf=0.0)

    assert ref_output.shape == (5, 930, 1024)
    assert fused_output.shape == ref_output.shape
    assert fused_output.dtype == ref_output.dtype
    assert torch.allclose(ref_output, fused_output, atol=1e-2, rtol=1e-2)

def test_attention_fused_forward_matches_reference_under_autocast():
    device = _require_npu()
    ref_attn, fused_attn = _build_attention_pair(device)

    torch.manual_seed(42)
    x = torch.randn(5, 930, 1024, device=device, dtype=torch.float32)
    pos = _build_positions(batch_size=5, seq_len=930, device=device)

    with torch.no_grad():
        with torch.npu.amp.autocast(dtype=torch.bfloat16):
            ref_output = ref_attn(x, pos=pos)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            fused_output = fused_attn(x, pos=pos)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()

    ref_output = torch.nan_to_num(ref_output, nan=0.0, posinf=0.0, neginf=0.0)
    fused_output = torch.nan_to_num(fused_output, nan=0.0, posinf=0.0, neginf=0.0)

    assert ref_output.shape == fused_output.shape
    assert ref_output.dtype == fused_output.dtype
    assert torch.allclose(ref_output, fused_output, atol=2e-2, rtol=2e-2)


def test_attention_fused_supports_aclgraph_capture():
    device = _require_npu()
    torch.manual_seed(42)

    block = Block(
        dim=1024,
        num_heads=16,
        mlp_ratio=2.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        attn_class=Attention_fused,
        rope=RotaryPositionEmbedding2D(frequency=100),
    ).to(device=device, dtype=torch.float32)
    block.eval()

    runner = ACLGraphBlockRunner(GraphConfig(enabled=True, debug=True, shared_pool=False))
    x = torch.randn(2, 128, 1024, device=device, dtype=torch.float32)
    pos = _build_positions(batch_size=2, seq_len=128, device=device)

    with torch.no_grad():
        ref_output = block(x, pos=pos)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        graph_output = runner.run(block, x, pos, block_kind="frame", block_idx=0)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()

    assert len(runner.cache) == 1
    assert graph_output.shape == ref_output.shape
    assert torch.allclose(ref_output, graph_output, atol=2e-2, rtol=2e-2)

def _build_mlp_pair(device: torch.device):
    torch.manual_seed(42)

    ref_mlp = Mlp(
        in_features=1024,
        hidden_features=4096,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop=0.0,
        bias=True,
    ).to(device=device, dtype=torch.float32)

    fused_mlp = Mlp_fused(
        in_features=1024,
        hidden_features=4096,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop=0.0,
        bias=True,
    ).to(device=device, dtype=torch.float32)

    fused_mlp.load_state_dict(ref_mlp.state_dict(), strict=True)
    ref_mlp.eval()
    fused_mlp.eval()
    return ref_mlp, fused_mlp


def test_mlp_fused_state_dict_compatible():
    device = _require_npu()
    ref_mlp, fused_mlp = _build_mlp_pair(device)

    ref_keys = set(ref_mlp.state_dict().keys())
    fused_keys = set(fused_mlp.state_dict().keys())
    assert ref_keys == fused_keys


def test_mlp_fused_forward_matches_reference():
    device = _require_npu()
    ref_mlp, fused_mlp = _build_mlp_pair(device)

    torch.manual_seed(42)
    x = torch.randn(5, 930, 1024, device=device, dtype=torch.float32)

    with torch.no_grad():
        ref_output = ref_mlp(x)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        fused_output = fused_mlp(x)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()

    ref_output = torch.nan_to_num(ref_output, nan=0.0, posinf=0.0, neginf=0.0)
    fused_output = torch.nan_to_num(fused_output, nan=0.0, posinf=0.0, neginf=0.0)

    assert ref_output.shape == fused_output.shape
    assert ref_output.dtype == fused_output.dtype
    assert torch.allclose(ref_output, fused_output, atol=2e-2, rtol=2e-2)


def test_mlp_fused_forward_matches_reference_under_autocast():
    device = _require_npu()
    ref_mlp, fused_mlp = _build_mlp_pair(device)

    torch.manual_seed(42)
    x = torch.randn(5, 930, 1024, device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.npu.amp.autocast(dtype=torch.bfloat16):
            ref_output = ref_mlp(x)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            fused_output = fused_mlp(x)
            if hasattr(torch, "npu"):
                torch.npu.synchronize()

    ref_output = torch.nan_to_num(ref_output, nan=0.0, posinf=0.0, neginf=0.0)
    fused_output = torch.nan_to_num(fused_output, nan=0.0, posinf=0.0, neginf=0.0)

    assert ref_output.shape == fused_output.shape
    assert ref_output.dtype == fused_output.dtype
    assert torch.allclose(ref_output, fused_output, atol=3e-2, rtol=3e-2)

def test_mlp_fused_rejects_dropout():
    with pytest.raises(ValueError, match="dropout"):
        Mlp_fused(
            in_features=1024,
            hidden_features=4096,
            out_features=None,
            act_layer=torch.nn.GELU,
            drop=0.1,
            bias=True,
        )

