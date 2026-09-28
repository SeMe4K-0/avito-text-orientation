"""Датасеты: ровный кроп → (случайный) поворот на 180° → холсты + метка.

Источник всегда отдаёт ровную картинку, поворот делаем сами, поэтому метка известна точно.
"""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .preprocess import to_canvas
from .synth import SynthRenderer

ROTATE_180 = Image.Transpose.ROTATE_180


class SynthSource:
    """Синтетический кроп однозначно задаётся (seed, epoch, index): каждая эпоха — новые картинки."""

    def __init__(self, renderer: SynthRenderer, size: int, seed: int):
        self.renderer, self.size, self.seed = renderer, size, seed

    def __len__(self):
        return self.size

    def __call__(self, i: int, epoch: int = 0) -> Image.Image:
        return self.renderer(np.random.default_rng([self.seed, epoch, i]))


class FileSource:
    def __init__(self, paths: list[Path]):
        self.paths = [str(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __call__(self, i: int, epoch: int = 0) -> Image.Image:
        return Image.open(self.paths[i]).convert("RGB")


def augment_real(rng: np.random.Generator, img: Image.Image) -> Image.Image:
    """Лёгкие искажения реальных кропов: они и так из домена теста."""
    if rng.random() < 0.3:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.3:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.3:
        scale = rng.uniform(0.4, 1.0)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.BILINEAR)
    if rng.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(30, 95)))
        img = Image.open(buf).convert("RGB")
    return img


class OrientDataset(Dataset):
    """Элемент: (холст для каждой высоты из `heights`..., метка).

    rotation:
      "random" — поворот заново разыгрывается каждую эпоху (обучение);
      "fixed"  — поворот зависит только от индекса (валидация);
      "none"   — без поворота (разметка ровных кропов учителем).
    soft — вероятность учителя, что ровный кроп выглядит перевёрнутым; по умолчанию 0.
    """

    def __init__(self, source, heights: tuple[int, ...], seed: int, rotation: str = "random",
                 soft: np.ndarray | None = None, augment: bool = False):
        self.source, self.heights, self.seed = source, heights, seed
        self.rotation, self.augment = rotation, augment
        self.soft = np.zeros(len(source), np.float32) if soft is None else soft.astype(np.float32)
        self.epoch = 0

    def __len__(self):
        return len(self.source)

    def image(self, i: int) -> tuple[Image.Image, float]:
        """Кроп после аугментаций и поворота и его метка."""
        key = [self.seed, i] if self.rotation != "random" else [self.seed, self.epoch, i]
        rng = np.random.default_rng(key)
        img = self.source(i, self.epoch if self.rotation == "random" else 0)
        if self.augment:
            img = augment_real(rng, img)
        rotate = self.rotation != "none" and rng.random() < 0.5
        if rotate:
            img = img.transpose(ROTATE_180)
        # У повёрнутого кропа вероятность «перевёрнут» зеркальна исходной.
        return img, float(1.0 - self.soft[i] if rotate else self.soft[i])

    def __getitem__(self, i):
        img, y = self.image(i)
        canvases = [torch.from_numpy(to_canvas(img, h)) for h in self.heights]
        return (*canvases, torch.tensor([y], dtype=torch.float32))


class FolderDataset(Dataset):
    """Кропы без меток (тест): только холст."""

    def __init__(self, paths: list[Path], height: int):
        self.paths, self.height = [str(p) for p in paths], height

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return torch.from_numpy(to_canvas(img, self.height))
