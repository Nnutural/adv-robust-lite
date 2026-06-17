from __future__ import annotations

from collections.abc import Callable


def eot_grad(model, x, y, loss_fn: Callable, eot_samples: int):
    import torch

    samples = max(1, int(eot_samples))
    grads = []
    for _ in range(samples):
        x_sample = x.detach().requires_grad_(True)
        loss = loss_fn(model(x_sample), y)
        grads.append(torch.autograd.grad(loss, x_sample, only_inputs=True)[0])
    return torch.stack(grads, dim=0).mean(dim=0)
