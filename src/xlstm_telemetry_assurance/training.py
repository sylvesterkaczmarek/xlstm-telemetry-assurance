from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AdaptationResult:
    accepted: bool
    guard_loss_before: float
    guard_loss_after: float


def _nll(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.clamp(std**2, min=1e-8)
    return torch.mean(0.5 * torch.log(2 * torch.pi * var) + 0.5 * (target - mean) ** 2 / var)


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 10,
    lr: float = 3e-3,
) -> list[float]:
    model.train()
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mean, std = model(x)
        loss = _nll(mean, std, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    mean, std = model(torch.from_numpy(x))
    return mean.cpu().numpy(), std.cpu().numpy()


def _loss(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        mean, std = model(torch.from_numpy(x))
        loss = _nll(mean, std, torch.from_numpy(y))
    return float(loss)


def guarded_adaptation(
    model: nn.Module,
    adapt_x: np.ndarray,
    adapt_y: np.ndarray,
    guard_x: np.ndarray,
    guard_y: np.ndarray,
    steps: int = 10,
    lr: float = 1e-2,
    tolerance: float = 0.01,
) -> AdaptationResult:
    before = _loss(model, guard_x, guard_y)
    saved = deepcopy(model.state_dict())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    model.train()
    x = torch.from_numpy(adapt_x)
    y = torch.from_numpy(adapt_y)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        mean, std = model(x)
        loss = _nll(mean, std, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        optimizer.step()

    after = _loss(model, guard_x, guard_y)
    accepted = after <= before * (1.0 + tolerance)
    if not accepted:
        model.load_state_dict(saved)

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    return AdaptationResult(accepted=accepted, guard_loss_before=before, guard_loss_after=after)
