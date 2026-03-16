import pytest
import torch

from vggt.layers.attention import Attention_fused
from vggt.layers.rope import RotaryPositionEmbedding2D


def _require_aclgraph_npu() -> None:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pytest.skip("torch_npu is not available")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")

    if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
        pytest.skip("ACLGraph APIs are not available")


def _build_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(seq_len, device=device, dtype=torch.int64)
    y = token_ids // 8
    x = token_ids % 8
    pos = torch.stack((y, x), dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1).clone()


def test_attention_fused_torch_npu_ops_can_capture_raw_aclgraph():
    _require_aclgraph_npu()

    device = torch.device("npu")
    torch.manual_seed(42)
    attn = Attention_fused(
        dim=64,
        num_heads=4,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=True,
        fused_attn=True,
        rope=RotaryPositionEmbedding2D(frequency=10),
    ).to(device=device, dtype=torch.float32)
    attn.eval()

    x = torch.randn(2, 17, 64, device=device, dtype=torch.float32)
    pos = _build_positions(batch_size=2, seq_len=17, device=device)

    with torch.no_grad():
        ref = attn(x, pos=pos)
        torch.npu.synchronize()

        static_x = x.detach().clone()
        static_pos = pos.detach().clone()
        graph = torch.npu.NPUGraph()
        max_position = int(pos.max().item()) + 1
        attn.rope.set_max_position_override(max_position)

        try:
            with torch.npu.graph(graph, auto_dispatch_capture=True):
                out = attn(static_x, pos=static_pos)

            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
        finally:
            attn.rope.set_max_position_override(None)

    assert out.shape == ref.shape
    assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3)


def test_npu_rotary_mul_can_capture_raw_aclgraph():
    _require_aclgraph_npu()
    import torch_npu

    device = torch.device("npu")
    torch.manual_seed(0)
    tokens = torch.randn(2, 4, 17, 8, device=device, dtype=torch.float32)
    cos = torch.randn(2, 1, 17, 8, device=device, dtype=torch.float32).contiguous()
    sin = torch.randn(2, 1, 17, 8, device=device, dtype=torch.float32).contiguous()

    with torch.no_grad():
        ref = torch_npu.npu_rotary_mul(tokens, cos, sin, rotary_mode="half")
        torch.npu.synchronize()

        static_tokens = tokens.detach().clone()
        static_cos = cos.detach().clone()
        static_sin = sin.detach().clone()
        graph = torch.npu.NPUGraph()

        with torch.npu.graph(graph, auto_dispatch_capture=True):
            out = torch_npu.npu_rotary_mul(
                static_tokens,
                static_cos,
                static_sin,
                rotary_mode="half",
            )

        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()

    assert out.shape == ref.shape
    assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3)


def test_npu_fusion_attention_can_capture_raw_aclgraph():
    _require_aclgraph_npu()
    import torch_npu

    device = torch.device("npu")
    torch.manual_seed(0)
    q = torch.randn(2, 4, 17, 16, device=device, dtype=torch.float32).contiguous()
    k = torch.randn(2, 4, 17, 16, device=device, dtype=torch.float32).contiguous()
    v = torch.randn(2, 4, 17, 16, device=device, dtype=torch.float32).contiguous()

    with torch.no_grad():
        ref = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            4,
            "BNSD",
            scale=float(16 ** -0.5),
            keep_prob=1.0,
        )[0]
        torch.npu.synchronize()

        static_q = q.detach().clone()
        static_k = k.detach().clone()
        static_v = v.detach().clone()
        graph = torch.npu.NPUGraph()

        with torch.npu.graph(graph, auto_dispatch_capture=True):
            out = torch_npu.npu_fusion_attention(
                static_q,
                static_k,
                static_v,
                4,
                "BNSD",
                scale=float(16 ** -0.5),
                keep_prob=1.0,
            )[0]

        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()

    assert out.shape == ref.shape
    assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3)

