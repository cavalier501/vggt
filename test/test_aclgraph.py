import pytest
from functools import partial
import torch

from vggt.graph import ACLGraphBlockRunner, GraphConfig
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.models.aggregator import Aggregator


def _require_aclgraph_npu() -> torch.device:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        pytest.skip("torch_npu is not available")

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")

    if not hasattr(torch.npu, "NPUGraph") or not hasattr(torch.npu, "graph"):
        pytest.skip("ACLGraph APIs are not available")

    return torch.device("npu")


def _build_positions(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(seq_len, device=device, dtype=torch.int64)
    y = token_ids // 8
    x = token_ids % 8
    pos = torch.stack((y, x), dim=-1)
    return pos.unsqueeze(0).expand(batch_size, -1, -1).clone()


def _build_block(device: torch.device) -> Block:
    torch.manual_seed(42)
    block = Block(
        dim=64,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope=RotaryPositionEmbedding2D(frequency=10),
        fused_attn=False,
    ).to(device=device, dtype=torch.float32)
    block.eval()
    return block


def _build_aggregator(device: torch.device) -> Aggregator:
    torch.manual_seed(42)
    model = Aggregator(
        img_size=28,
        patch_size=14,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_register_tokens=2,
        patch_embed="conv",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=10,
        init_values=0.01,
        block_fn=partial(Block, fused_attn=False),
    ).to(device=device, dtype=torch.float32)
    model.eval()
    return model


def test_aclgraph_block_matches_eager():
    device = _require_aclgraph_npu()
    block = _build_block(device)
    runner = ACLGraphBlockRunner(GraphConfig(enabled=True, debug=True))

    torch.manual_seed(42)
    x = torch.randn(2, 17, 64, device=device, dtype=torch.float32)
    pos = _build_positions(batch_size=2, seq_len=17, device=device)

    with torch.no_grad():
        ref = block(x, pos=pos)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        out = runner.run(block, x, pos, block_kind="frame", block_idx=0)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    assert len(runner.cache) == 1
    assert out.shape == ref.shape
    assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3)


def test_aclgraph_runner_replays_same_shape_and_caches_new_shape():
    device = _require_aclgraph_npu()
    block = _build_block(device)
    runner = ACLGraphBlockRunner(GraphConfig(enabled=True, debug=True))

    torch.manual_seed(7)
    x = torch.randn(2, 17, 64, device=device, dtype=torch.float32)
    pos = _build_positions(batch_size=2, seq_len=17, device=device)

    with torch.no_grad():
        runner.run(block, x, pos, block_kind="frame", block_idx=0)
        runner.run(block, x, pos, block_kind="frame", block_idx=0)

    assert len(runner.cache) == 1
    entry = next(iter(runner.cache.values()))
    assert entry.replay_count >= 1

    x2 = torch.randn(3, 17, 64, device=device, dtype=torch.float32)
    pos2 = _build_positions(batch_size=3, seq_len=17, device=device)

    with torch.no_grad():
        runner.run(block, x2, pos2, block_kind="frame", block_idx=0)

    assert len(runner.cache) == 2


def test_aggregator_aclgraph_matches_eager():
    device = _require_aclgraph_npu()
    eager_model = _build_aggregator(device)
    graph_model = _build_aggregator(device)
    graph_model.load_state_dict(eager_model.state_dict(), strict=True)
    graph_model.enable_graph(GraphConfig(enabled=True, debug=True))

    torch.manual_seed(21)
    images = torch.rand(1, 2, 3, 28, 28, device=device, dtype=torch.float32)

    with torch.no_grad():
        eager_out, eager_idx = eager_model(images)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()
        graph_out, graph_idx = graph_model(images)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    assert graph_idx == eager_idx
    assert len(graph_out) == len(eager_out)
    assert graph_model._graph_runner is not None
    assert len(graph_model._graph_runner.cache) > 0

    for ref, out in zip(eager_out, graph_out):
        assert ref.shape == out.shape
        assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3)


def test_aggregator_aclgraph_uses_shared_pool_when_available():
    device = _require_aclgraph_npu()
    model = _build_aggregator(device)
    model.enable_graph(GraphConfig(enabled=True, shared_pool=True, debug=True))

    runner = model._graph_runner
    assert runner is not None


    torch.manual_seed(123)
    images = torch.rand(1, 2, 3, 28, 28, device=device, dtype=torch.float32)

    with torch.no_grad():
        model(images)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    assert runner.uses_shared_pool
    assert len(runner.cache) > 0
    assert runner.graph_pool is not None
    assert runner.capture_with_pool_count > 0



def test_aggregator_clear_graph_cache_empties_runner_cache():
    device = _require_aclgraph_npu()
    model = _build_aggregator(device)
    model.enable_graph(GraphConfig(enabled=True, shared_pool=True, debug=True))

    runner = model._graph_runner
    assert runner is not None

    torch.manual_seed(321)
    images = torch.rand(1, 2, 3, 28, 28, device=device, dtype=torch.float32)

    with torch.no_grad():
        model(images)
        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

    assert len(runner.cache) > 0
    model.clear_graph_cache()
    assert len(runner.cache) == 0
    assert runner.capture_with_pool_count == 0
def test_aggregator_disable_graph_restores_eager_path():
    device = _require_aclgraph_npu()
    model = _build_aggregator(device)
    model.enable_graph(GraphConfig(enabled=True, debug=True))
    model.disable_graph()

    torch.manual_seed(99)
    images = torch.rand(1, 2, 3, 28, 28, device=device, dtype=torch.float32)

    with torch.no_grad():
        out, patch_start_idx = model(images)

    assert model._graph_runner is None
    assert isinstance(out, list)
    assert len(out) == model.depth
    assert patch_start_idx == model.patch_start_idx



