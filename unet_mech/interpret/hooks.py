from __future__ import annotations

from typing import Dict, List, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn
from loguru import logger


@runtime_checkable
class _HookableModel(Protocol):
    """Models that expose a static hook list (e.g. ResNet18UNet)."""

    _activations: Dict[str, torch.Tensor]
    _hooks: List

    def hook_target_layers(self) -> List[Tuple[str, nn.Module]]: ...


def clear_activations(model: nn.Module) -> None:
    if hasattr(model, "_activations"):
        model._activations.clear()


def register_hooks(
    model: nn.Module,
    named_modules: List[Tuple[str, nn.Module]] | None = None,
) -> None:
    """
    Attach forward hooks. If named_modules is None, model.hook_target_layers()
    is used (ResNet18UNet and future Baby U-Net should implement it).
    """
    if not hasattr(model, "_activations"):
        model._activations = {}
    if not hasattr(model, "_hooks"):
        model._hooks = []
    remove_hooks(model)
    clear_activations(model)

    if named_modules is None:
        if not isinstance(model, _HookableModel):
            raise TypeError("model must implement hook_target_layers() or pass named_modules")
        named_modules = model.hook_target_layers()

    def _make_hook(name: str):
        def hook(module, inp, output):
            model._activations[name] = output.detach().cpu()

        return hook

    for name, mod in named_modules:
        h = mod.register_forward_hook(_make_hook(name))
        model._hooks.append(h)


def remove_hooks(model: nn.Module) -> None:
    if not hasattr(model, "_hooks"):
        return
    for h in model._hooks:
        h.remove()
    model._hooks.clear()


def get_activations(model: nn.Module) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "_activations"):
        return {}
    return dict(model._activations)


def print_architecture_reference(
    model: nn.Module,
    img_size: int,
    device: str,
    model_title: str = "ResNet18-Encoder U-Net",
    named_modules: List[Tuple[str, nn.Module]] | None = None,
    input_channels: int = 3,
) -> None:
    register_hooks(model, named_modules=named_modules)
    dummy = torch.zeros(1, input_channels, img_size, img_size, device=device)
    with torch.no_grad():
        _ = model(dummy)
    acts = get_activations(model)
    remove_hooks(model)

    logger.info("\n" + "=" * 65)
    logger.info(f" Architecture Reference — {model_title}")
    logger.info("=" * 65)
    for name, tensor in acts.items():
        b, c, h, w = tensor.shape
        tag = " ← bottleneck" if name == "enc_layer4" else ""
        logger.info(f"  {name:<14} [{b}, {c:>3}, {h:>3}, {w:>3}]{tag}")

    logger.info("-" * 65)
    if hasattr(model, "encoder_parameters") and hasattr(model, "decoder_parameters"):
        n_enc = sum(p.numel() for p in model.encoder_parameters())
        n_dec = sum(p.numel() for p in model.decoder_parameters())
        logger.info(f"  Encoder params : {n_enc:>10,}")
        logger.info(f"  Decoder params : {n_dec:>10,}")
        logger.info(f"  Total params   : {n_enc + n_dec:>10,}")
    else:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total params   : {n_params:>10,}")
    logger.info("=" * 65 + "\n")
