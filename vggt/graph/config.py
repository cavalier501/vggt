from dataclasses import dataclass


@dataclass(slots=True)
class GraphConfig:
    enabled: bool = False
    backend: str = "aclgraph"
    scope: str = "block"
    debug: bool = False
    force_eager_sdpa: bool = False
    shared_pool: bool = False


