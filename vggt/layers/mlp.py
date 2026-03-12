# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_fused(nn.Module):
    """
    Fused FFN implementation for Aggregator inference blocks.

    This class is currently not enabled in production because performance
    validation did not show a clear benefit.

    The evaluated inference path uses:
    - hidden_features=4096
    - out_features=None (effective output dim is 1024)
    - act_layer=nn.GELU
    - drop=0.0
    - bias=True

    Unsupported in this fused path:
    - dropout (drop must be 0.0)
    - activations other than nn.GELU
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if drop != 0.0:
            raise ValueError("Mlp_fused does not support dropout; expected drop=0.0")
        if act_layer is not nn.GELU:
            raise ValueError("Mlp_fused only supports nn.GELU")

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        # These cached tensors are populated on the first forward() or after a device move,
        # so they cannot be fixed once in __init__. This implementation does not detect
        # state_dict changes after the cache has been built; if weights are updated later,
        # the fused cache must be refreshed manually.
        self._fused_param_device = None
        self.weight1 = None
        self.weight2 = None
        self.bias1 = None
        self.bias2 = None

    def _refresh_fused_params(self, device) -> None:
        import torch

        self.weight1 = self.fc1.weight.transpose(0, 1).contiguous().to(device=device, dtype=torch.bfloat16)
        self.weight2 = self.fc2.weight.transpose(0, 1).contiguous().to(device=device, dtype=torch.bfloat16)
        self.bias1 = self.fc1.bias.contiguous().to(device=device, dtype=torch.float32) if self.fc1.bias is not None else None
        self.bias2 = self.fc2.bias.contiguous().to(device=device, dtype=torch.float32) if self.fc2.bias is not None else None
        self._fused_param_device = device

    def forward(self, x: Tensor) -> Tensor:
        import torch
        import torch_npu

        original_dtype = x.dtype
        if self.weight1 is None or self._fused_param_device != x.device:
            self._refresh_fused_params(x.device)

        x_input = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        x = torch_npu.npu_ffn(
            x_input,
            self.weight1,
            self.weight2,
            "gelu",
            bias1=self.bias1,
            bias2=self.bias2,
            inner_precise=0,
        )
        if hasattr(torch, "npu") and torch.npu.is_autocast_enabled():
            return x
        return x if x.dtype == original_dtype else x.to(original_dtype)

