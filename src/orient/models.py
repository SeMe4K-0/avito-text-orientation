"""Учитель и студент: сети timm с одноканальным входом и одним логитом «перевёрнут»."""
import timm
import torch

TEACHER = {"name": "convnext_tiny.fb_in22k_ft_in1k", "height": 48}
STUDENT = {"name": "lcnet_050.ra2_in1k", "height": 32}


def build(name: str, pretrained: bool = True) -> torch.nn.Module:
    # in_chans=1: timm суммирует веса первой свёртки по RGB, предобучение сохраняется.
    return timm.create_model(name, pretrained=pretrained, in_chans=1, num_classes=1)


def tta_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Антисимметричный TTA: (f(x) − f(rot180 x)) / 2, поэтому p(x) + p(rot180 x) = 1 точно."""
    z = model(torch.cat([x, torch.flip(x, dims=(2, 3))]))
    a, b = z.chunk(2)
    return (a - b) / 2


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
