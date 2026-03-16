from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .config import GraphConfig

logger = logging.getLogger(__name__)


GraphCacheKey = Tuple[
    str,
    int,
    Tuple[int, ...],
    torch.dtype,
    Optional[Tuple[int, ...]],
    Optional[torch.dtype],
    str,
]


@dataclass
class ACLGraphEntry:
    block: nn.Module
    static_x: torch.Tensor
    static_pos: Optional[torch.Tensor]
    graph: "torch.npu.NPUGraph"
    output: torch.Tensor
    capture_count: int = 1
    replay_count: int = 0


class ACLGraphBlockRunner:
    def __init__(self, config: GraphConfig):
        self.config = config
        self.cache: Dict[GraphCacheKey, ACLGraphEntry] = {}
        self.graph_pool: Optional[Any] = self._init_graph_pool()
        self.uses_shared_pool: bool = self.graph_pool is not None
        self.capture_with_pool_count: int = 0

    @staticmethod
    def _npu_available() -> bool:
        return (
            hasattr(torch, "npu")
            and callable(getattr(torch.npu, "is_available", None))
            and torch.npu.is_available()
            and hasattr(torch.npu, "NPUGraph")
            and hasattr(torch.npu, "graph")
        )

    def _init_graph_pool(self) -> Optional[Any]:
        if not self.config.shared_pool:
            return None
        return torch.npu.graph_pool_handle()

    def is_enabled(self) -> bool:
        return (
            self.config.enabled
            and self.config.backend == "aclgraph"
            and self.config.scope == "block"
            and self._npu_available()
        )

    def clear_cache(self) -> None:
        self.cache.clear()
        self.capture_with_pool_count = 0
        self.graph_pool = self._init_graph_pool()
        self.uses_shared_pool = self.graph_pool is not None
        if self._npu_available() and hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()

    @staticmethod
    def _resolve_max_position(pos: Optional[torch.Tensor]) -> Optional[int]:
        if pos is None:
            return None
        return int(pos.max().item()) + 1

    @staticmethod
    def _set_rope_override(block: nn.Module, max_position: Optional[int]) -> None:
        for module in block.modules():
            if hasattr(module, "set_max_position_override"):
                module.set_max_position_override(max_position)

    @staticmethod
    def _set_graph_capture_mode(block: nn.Module, enabled: bool) -> None:
        for module in block.modules():
            if hasattr(module, "set_graph_capture_mode"):
                module.set_graph_capture_mode(enabled)

    def _make_key(
        self,
        block_kind: str,
        block_idx: int,
        x: torch.Tensor,
        pos: Optional[torch.Tensor],
    ) -> GraphCacheKey:
        return (
            block_kind,
            block_idx,
            tuple(x.shape),
            x.dtype,
            None if pos is None else tuple(pos.shape),
            None if pos is None else pos.dtype,
            str(x.device),
        )

    def run(
        self,
        block: nn.Module,
        x: torch.Tensor,
        pos: Optional[torch.Tensor],
        block_kind: str,
        block_idx: int,
    ) -> torch.Tensor:
        if not self.is_enabled():
            return block(x, pos=pos)

        key = self._make_key(block_kind, block_idx, x, pos)
        entry = self.cache.get(key)
        if entry is None:
            try:
                entry = self._capture(block, x, pos)
            except Exception:
                if self.config.debug:
                    logger.exception(
                        "ACLGraph capture failed for %s block %s. Falling back to eager.",
                        block_kind,
                        block_idx,
                    )
                return block(x, pos=pos)
            self.cache[key] = entry

        try:
            entry.static_x.copy_(x)
            if pos is not None and entry.static_pos is not None:
                entry.static_pos.copy_(pos)
            entry.graph.replay()
            entry.replay_count += 1
            return entry.output
        except Exception:
            if self.config.debug:
                logger.exception(
                    "ACLGraph replay failed for %s block %s. Falling back to eager.",
                    block_kind,
                    block_idx,
                )
            return block(x, pos=pos)

    def _capture(
        self,
        block: nn.Module,
        x: torch.Tensor,
        pos: Optional[torch.Tensor],
    ) -> ACLGraphEntry:
        static_x = x.detach().clone()
        static_pos = None if pos is None else pos.detach().clone()
        max_position = self._resolve_max_position(pos)

        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        self._set_rope_override(block, max_position)
        self._set_graph_capture_mode(block, True)
        try:
            output = self._capture_graph(block, static_x, static_pos, graph)
        finally:
            self._set_graph_capture_mode(block, False)
            self._set_rope_override(block, None)

        if hasattr(torch.npu, "synchronize"):
            torch.npu.synchronize()

        return ACLGraphEntry(
            block=block,
            static_x=static_x,
            static_pos=static_pos,
            graph=graph,
            output=output,
        )

    def _capture_graph(
        self,
        block: nn.Module,
        static_x: torch.Tensor,
        static_pos: Optional[torch.Tensor],
        graph: "torch.npu.NPUGraph",
    ) -> torch.Tensor:
        if self.graph_pool is not None:
            with torch.npu.graph(graph, pool=self.graph_pool, auto_dispatch_capture=True):
                output = block(static_x, pos=static_pos)
            self.capture_with_pool_count += 1
            return output

        with torch.npu.graph(graph, auto_dispatch_capture=True):
            return block(static_x, pos=static_pos)

