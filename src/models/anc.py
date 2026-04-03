from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


def build_mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if depth < 1:
        raise ValueError("depth must be at least 1")

    layers: list[nn.Module] = []
    prev = input_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class ModelShape:
    plaintext_len: int
    key_len: int
    ciphertext_len: int


class AliceModel(nn.Module):
    def __init__(
        self,
        plaintext_len: int,
        key_len: int,
        ciphertext_len: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.shape = ModelShape(plaintext_len, key_len, ciphertext_len)
        self.net = build_mlp(plaintext_len + key_len, ciphertext_len, width, depth)

    def forward(self, plaintext: Tensor, key: Tensor) -> Tensor:
        if plaintext.shape != key.shape:
            raise ValueError(f"plaintext and key must share shape, got {plaintext.shape} vs {key.shape}")
        x = torch.cat([plaintext, key], dim=1)
        return self.net(x)


class BobModel(nn.Module):
    def __init__(
        self,
        plaintext_len: int,
        key_len: int,
        ciphertext_len: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.shape = ModelShape(plaintext_len, key_len, ciphertext_len)
        self.net = build_mlp(ciphertext_len + key_len, plaintext_len, width, depth)

    def forward(self, ciphertext: Tensor, key: Tensor) -> Tensor:
        x = torch.cat([ciphertext, key], dim=1)
        return self.net(x)


class EveModel(nn.Module):
    def __init__(
        self,
        plaintext_len: int,
        ciphertext_len: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.plaintext_len = plaintext_len
        self.ciphertext_len = ciphertext_len
        self.net = build_mlp(ciphertext_len, plaintext_len, width, depth)

    def forward(self, ciphertext: Tensor) -> Tensor:
        return self.net(ciphertext)


def hard_bits_from_logits(logits: Tensor) -> Tensor:
    return (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)


def bit_accuracy(pred_bits: Tensor, target_bits: Tensor) -> float:
    if pred_bits.shape != target_bits.shape:
        raise ValueError(f"Shape mismatch: {pred_bits.shape} != {target_bits.shape}")
    return float((pred_bits == target_bits).to(dtype=torch.float32).mean().item())


def bit_error_rate(pred_bits: Tensor, target_bits: Tensor) -> float:
    return 1.0 - bit_accuracy(pred_bits, target_bits)


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5
