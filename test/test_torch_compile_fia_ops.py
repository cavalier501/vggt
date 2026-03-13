import math

import pytest
import torch
from torch import nn

from vggt.layers.rope import RotaryPositionEmbedding2D


def _require_torchair_npu() -> None:
    try:
        import torch_npu  # noqa: F401
        import torchair  # noqa: F401
        from torchair.configs.compiler_config import CompilerConfig  # noqa: F401
    except Exception as exc:
        pytest.skip(f"torchair or torch_npu is not available: {exc}")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")


def _make_npu_backend():
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    return tng.get_npu_backend(compiler_config=config)


def _build_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(seq_len, device=device, dtype=torch.int64)
    y = token_ids // 8
    x = token_ids % 8
    pos = torch.stack((y, x), dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1).clone()


class _FIAOpModule(nn.Module):
    def __init__(self, num_heads: int, scale: float, actseqlen: list[int], actseqlenkv: list[int]) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.actseqlen = actseqlen
        self.actseqlenkv = actseqlenkv

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        import torch_npu

        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            actual_seq_lengths=self.actseqlen,
            actual_seq_lengths_kv=self.actseqlenkv,
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale=self.scale,
            pre_tokens=65535,
            next_tokens=65535,
        )[0]


class CompileFIAModule(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = RotaryPositionEmbedding2D(frequency=10)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        import torch_npu

        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seqlen, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        attn_dtype = v.dtype
        if q.dtype != attn_dtype:
            q = q.to(attn_dtype)
        if k.dtype != attn_dtype:
            k = k.to(attn_dtype)

        out = torch_npu.npu_fused_infer_attention_score(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            actual_seq_lengths=[seqlen],
            actual_seq_lengths_kv=[seqlen],
            num_heads=self.num_heads,
            input_layout="BNSD",
            scale=self.scale,
            pre_tokens=65535,
            next_tokens=65535,
        )[0]
        out = out.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.proj(out)


def test_compile_npu_fused_infer_attention_score_matches_eager():
    _require_torchair_npu()

    device = torch.device("npu")
    dtype = torch.float16
    num_heads = 8
    q_len = 164
    kv_len = 1024
    head_dim = 128
    scale = 1.0 / math.sqrt(head_dim)
    backend = _make_npu_backend()

    torch.manual_seed(0)
    q = torch.randn(1, num_heads, q_len, head_dim, dtype=dtype, device=device).contiguous()
    k = torch.randn(1, num_heads, kv_len, head_dim, dtype=dtype, device=device).contiguous()
    v = torch.randn(1, num_heads, kv_len, head_dim, dtype=dtype, device=device).contiguous()

    module = _FIAOpModule(
        num_heads=num_heads,
        scale=scale,
        actseqlen=[q_len],
        actseqlenkv=[kv_len],
    ).to(device)
    module.eval()

    with torch.no_grad():
        eager = module(q, k, v)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

        compiled = torch.compile(module, backend=backend, dynamic=False, fullgraph=True)
        compiled_out = compiled(q, k, v)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    assert compiled_out.shape == eager.shape
    assert torch.allclose(eager.float(), compiled_out.float(), atol=1e-2, rtol=1e-2)


def test_compile_attention_fused_module_matches_eager():
    _require_torchair_npu()

    device = torch.device("npu")
    dtype = torch.float16
    backend = _make_npu_backend()

    torch.manual_seed(0)
    module = CompileFIAModule().to(device=device, dtype=dtype)
    module.eval()

    x = torch.randn(2, 17, 64, dtype=dtype, device=device)
    pos = _build_positions(batch_size=2, seq_len=17, device=device)
    max_position = int(pos.max().item()) + 1
    module.rope.set_max_position_override(max_position)
    try:
        with torch.no_grad():
            eager = module(x, pos)
            if hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()

            compiled = torch.compile(module, backend=backend, dynamic=False, fullgraph=True)
            compiled_out = compiled(x, pos)
            if hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
    finally:
        module.rope.set_max_position_override(None)

    assert compiled_out.shape == eager.shape
    assert torch.allclose(eager.float(), compiled_out.float(), atol=1e-2, rtol=1e-2)
