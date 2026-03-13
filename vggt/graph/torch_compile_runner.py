from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

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
class TorchCompileEntry:
    block: nn.Module
    compiled_block: Callable[..., torch.Tensor]
    max_position: Optional[int]
    compile_count: int = 1
    run_count: int = 0


class TorchCompileBlockRunner:
    def __init__(self, config: GraphConfig):
        self.config = config
        self.cache: Dict[GraphCacheKey, TorchCompileEntry] = {}
        self._npu_backend = self._init_backend()

    @staticmethod
    def _torch_compile_available() -> bool:
        return (
            hasattr(torch, "compile")
            and hasattr(torch, "npu")
            and callable(getattr(torch.npu, "is_available", None))
            and torch.npu.is_available()
        )

    def _init_backend(self) -> Any:
        if not self._torch_compile_available():
            return None
        try:
            import torchair as tng
            from torchair.configs.compiler_config import CompilerConfig
        except Exception:
            if self.config.debug:
                logger.exception("torchair backend is unavailable")
            return None

        config = CompilerConfig()
        return tng.get_npu_backend(compiler_config=config)

    def is_enabled(self) -> bool:
        return (
            self.config.enabled
            and self.config.backend == "torch_compile"
            and self.config.scope == "block"
            and self._npu_backend is not None
        )

    def clear_cache(self) -> None:
        for entry in self.cache.values():
            self._set_graph_backend(entry.block, "eager")
            self._set_rope_override(entry.block, None)
        self.cache.clear()

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
    def _set_graph_backend(block: nn.Module, mode: str) -> None:
        for module in block.modules():
            if hasattr(module, "set_graph_backend"):
                module.set_graph_backend(mode)

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
                entry = self._compile(block, pos)
            except Exception:
                self._set_graph_backend(block, "eager")
                self._set_rope_override(block, None)
                if self.config.debug:
                    logger.exception("torch.compile block compilation failed. Falling back to eager.")
                return block(x, pos=pos)
            self.cache[key] = entry

        self._set_graph_backend(entry.block, "torch_compile")
        self._set_rope_override(entry.block, entry.max_position)
        try:
            out = entry.compiled_block(x, pos)
            entry.run_count += 1
            return out
        except Exception:
            self._set_graph_backend(entry.block, "eager")
            self._set_rope_override(entry.block, None)
            if self.config.debug:
                logger.exception("torch.compile block execution failed. Falling back to eager.")
            return block(x, pos=pos)

    def _compile(
        self,
        block: nn.Module,
        pos: Optional[torch.Tensor],
    ) -> TorchCompileEntry:
        max_position = self._resolve_max_position(pos)
        self._set_graph_backend(block, "torch_compile")
        self._set_rope_override(block, max_position)
        try:
            compiled_block = torch.compile(
                block,
                backend=self._npu_backend,
                dynamic=False,
                fullgraph=True,
            )
        except Exception:
            self._set_graph_backend(block, "eager")
            self._set_rope_override(block, None)
            raise

        return TorchCompileEntry(
            block=block,
            compiled_block=compiled_block,
            max_position=max_position,
        )
