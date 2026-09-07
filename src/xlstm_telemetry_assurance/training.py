from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from numbers import Real
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AdaptationResult:
    accepted: bool
    guard_loss_before: float
    guard_loss_after: float


def _nll(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mean.ndim != 2 or mean.numel() == 0 or mean.shape != std.shape or mean.shape != target.shape:
        raise ValueError("mean, std and target must have the same nonempty (samples, channels) shape")
    if not all(bool(torch.isfinite(value).all()) for value in (mean, std, target)):
        raise FloatingPointError("Gaussian predictions and targets must be finite")
    if not bool((std > 0).all()):
        raise ValueError("predicted standard deviations must be positive")
    loss = torch.mean(torch.log(std) + 0.5 * math.log(2 * math.pi) + 0.5 * ((target - mean) / std) ** 2)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Gaussian loss is nonfinite")
    return loss


def _validate_array(name: str, value: np.ndarray, ndim: int) -> None:
    if not isinstance(value, np.ndarray) or value.ndim != ndim or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional NumPy array")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _validate_data(x: np.ndarray, y: np.ndarray | None = None) -> None:
    _validate_array("x", x, 3)
    if y is not None:
        _validate_array("y", y, 2)
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples")


def _validate_steps(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _validate_control(name: str, value: float, *, positive: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {requirement}")


def _tensor(model: nn.Module, value: np.ndarray) -> torch.Tensor:
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next((buffer for buffer in model.buffers() if buffer.is_floating_point()), None)
    tensor = (
        torch.as_tensor(value)
        if reference is None
        else torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    )
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("input values overflow the model's numerical precision")
    return tensor


def _checked_step(parameters: list[nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
    try:
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    except RuntimeError as error:
        raise FloatingPointError("gradient norm is nonfinite") from error
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise FloatingPointError("optimizer produced nonfinite parameters")


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int = 10,
    lr: float = 3e-3,
) -> list[float]:
    _validate_data(train_x, train_y)
    _validate_steps("epochs", epochs)
    _validate_control("lr", lr, positive=True)
    x = _tensor(model, train_x)
    y = _tensor(model, train_y)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mean, std = model(x)
        loss = _nll(mean, std, y)
        loss.backward()
        _checked_step(parameters, optimizer)
        losses.append(float(loss.detach()))
    return losses


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_data(x)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        mean, std = model(_tensor(model, x))
        if mean.ndim != 2 or mean.shape != std.shape or mean.shape[0] != len(x) or mean.numel() == 0:
            raise ValueError("predictions must have matching nonempty (samples, channels) shapes")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise FloatingPointError("predictions must be finite")
        if not bool((std > 0).all()):
            raise ValueError("predicted standard deviations must be positive")
        return mean.cpu().numpy(), std.cpu().numpy()
    finally:
        for module, training in modes:
            module.training = training


def _loss(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        mean, std = model(_tensor(model, x))
        loss = _nll(mean, std, _tensor(model, y))
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
    """Try head-only updates with a transactional guard check.

    Accept when ``after <= before + tolerance * abs(before)``. This permits
    relative degradation for either sign of Gaussian NLL; a zero baseline
    permits no degradation. On rejection or an exception, restore parameters
    and buffers. Always restore caller modes, gradient flags and gradients.
    Numerical failures raise instead of returning a nonfinite guard result.
    """
    _validate_data(adapt_x, adapt_y)
    _validate_data(guard_x, guard_y)
    _validate_steps("steps", steps)
    _validate_control("lr", lr, positive=True)
    _validate_control("tolerance", tolerance, positive=False)
    head = getattr(model, "head", None)
    if not isinstance(head, nn.Module):
        raise ValueError("guarded adaptation requires a model.head module")
    parameters = list(model.parameters())
    head_parameters = list(head.parameters())
    if not head_parameters:
        raise ValueError("model.head must have parameters to adapt")
    saved = deepcopy(model.state_dict())
    modes = [(module, module.training) for module in model.modules()]
    gradient_state = [(parameter, parameter.requires_grad, parameter.grad) for parameter in parameters]
    accepted = False
    try:
        before = _loss(model, guard_x, guard_y)
        for parameter in parameters:
            parameter.requires_grad_(False)
        for parameter in head_parameters:
            parameter.requires_grad_(True)

        optimizer = torch.optim.Adam(head_parameters, lr=lr)
        # Frozen feature extractors must not update running statistics or use
        # training-only stochastic layers while adapting the output head.
        model.eval()
        head.train()
        x = _tensor(model, adapt_x)
        y = _tensor(model, adapt_y)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            mean, std = model(x)
            loss = _nll(mean, std, y)
            loss.backward()
            _checked_step(head_parameters, optimizer)

        after = _loss(model, guard_x, guard_y)
        accepted = after <= before + tolerance * abs(before)
        return AdaptationResult(accepted=accepted, guard_loss_before=before, guard_loss_after=after)
    finally:
        if not accepted:
            model.load_state_dict(saved)
        for module, training in modes:
            module.training = training
        for parameter, requires_grad, gradient in gradient_state:
            parameter.requires_grad_(requires_grad)
            parameter.grad = gradient
