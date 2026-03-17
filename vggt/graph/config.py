from dataclasses import dataclass


@dataclass(slots=True)
class GraphConfig:
    enabled: bool = False
    backend: str = "torch_compile"
    scope: str = "block"
    debug: bool = False
    force_eager_sdpa: bool = False
    shared_pool: bool = False
