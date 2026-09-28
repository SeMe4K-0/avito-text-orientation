"""Рендер синтетических строк, похожих на кропы детектора текста.

Метка ориентации у синтетики известна: строка рисуется ровно, поворот делает датасет.
"""
import io
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .text import TextSampler

# Шрифт берём, только если в нём есть весь базовый алфавит: иначе вместо букв рисуются квадраты.
REQUIRED_CHARS = ("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def font_charset(path: Path) -> frozenset[int]:
    return frozenset(TTFont(path, fontNumber=0, lazy=True).getBestCmap() or {})


def collect_fonts(root: Path) -> dict[str, list[tuple[str, frozenset[int]]]]:
    """Семейство (папка) → список (путь, набор символов) для шрифтов с кириллицей и латиницей."""
    families = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".ttf", ".otf"):
            continue
        try:
            charset = font_charset(p)
        except Exception:
            continue
        if all(ord(c) in charset for c in REQUIRED_CHARS):
            families[p.parent.name].append((str(p), charset))
    return dict(families)


@lru_cache(maxsize=1024)
def _font(path: str, size: int, variation: bytes | None) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size)
    if variation is not None:
        font.set_variation_by_name(variation)
    return font


@lru_cache(maxsize=256)
def _variations(path: str) -> tuple[bytes, ...]:
    try:
        return tuple(ImageFont.truetype(path, 10).get_variation_names())
    except OSError:
        return ()


def _luma(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _colors(rng) -> tuple[np.ndarray, np.ndarray]:
    """Фон и текст. Чаще всего тёмный текст на светлом фоне, как на скриншотах и этикетках."""
    r = rng.random()
    if r < 0.35:
        bg = rng.integers(215, 256, 3)
        fg = rng.integers(0, 70, 3)
    elif r < 0.5:
        bg = rng.integers(0, 60, 3)
        fg = rng.integers(200, 256, 3)
    else:
        bg = _muted(rng)
        fg = _muted(rng)
        for _ in range(20):
            if abs(_luma(fg) - _luma(bg)) > 80:
                break
            fg = _muted(rng)
        else:
            fg = 255 - bg
    return bg.astype(np.float32), fg.astype(np.float32)


def _muted(rng) -> np.ndarray:
    """Случайный цвет с пониженной насыщенностью: кислотные цвета в тесте редкость."""
    c = rng.integers(0, 256, 3).astype(np.float32)
    gray = _luma(c)
    return gray + (c - gray) * rng.uniform(0.15, 1.0)


def _background(rng, h: int, w: int, color: np.ndarray) -> np.ndarray:
    img = np.broadcast_to(color, (h, w, 3)).astype(np.float32)
    kind = rng.random()
    if kind < 0.25:
        # Градиент в случайную сторону: направление не должно подсказывать ориентацию.
        other = np.clip(color + rng.normal(0, 50, 3), 0, 255)
        axis = np.linspace(0, 1, w if rng.random() < 0.5 else h, dtype=np.float32)
        t = axis[None, :, None] if len(axis) == w else axis[:, None, None]
        img = img * (1 - t) + other * t
    elif kind < 0.5:
        small = rng.normal(0, rng.uniform(8, 30), (max(2, h // 8), max(2, w // 8))).astype(np.float32)
        noise = np.asarray(Image.fromarray(small, "F").resize((w, h), Image.BICUBIC))
        img = img + noise[..., None]
    return img


class SynthRenderer:
    def __init__(self, fonts_root: Path, sampler: TextSampler):
        self.families = list(collect_fonts(fonts_root).values())
        if not self.families:
            raise RuntimeError(f"В {fonts_root} нет шрифтов с кириллицей")
        self.sampler = sampler

    def _pick_font(self, rng):
        fonts = self.families[rng.integers(len(self.families))]
        return fonts[rng.integers(len(fonts))]

    def _text(self, rng, charset) -> str:
        for _ in range(10):
            text = self.sampler.sample(rng)
            if text.strip() and all(ord(c) in charset for c in text if not c.isspace()):
                return text
        return "".join(rng.choice(list(REQUIRED_CHARS), rng.integers(3, 12)))

    def __call__(self, rng: np.random.Generator) -> Image.Image:
        path, charset = self._pick_font(rng)
        names = _variations(path)
        variation = names[rng.integers(len(names))] if names else None
        size = int(rng.integers(28, 72))
        font = _font(path, size, variation)

        text = self._text(rng, charset)
        length = int(font.getlength(text)) + 1
        width = length + 4 * size
        height = 7 * size
        baseline = height // 2 + size // 3
        x0 = 2 * size

        stroke_w = int(rng.integers(1, max(2, size // 12) + 1)) if rng.random() < 0.1 else 0
        fill = Image.new("L", (width, height))
        stroke = Image.new("L", (width, height)) if stroke_w else None
        main = Image.new("L", (width, height))

        # Куски соседних строк сверху/снизу: в тесте они часто попадают в кроп.
        lines = [(baseline, text, True)]
        spacing = size * rng.uniform(1.05, 1.6)
        for sign in (-1, 1):
            if rng.random() < 0.3:
                lines.append((int(baseline + sign * spacing), self._text(rng, charset), False))

        for y, line, is_main in lines:
            x = x0 if is_main else x0 + int(rng.integers(-size, size + 1))
            ImageDraw.Draw(fill).text((x, y), line, font=font, fill=255, anchor="ls")
            if stroke:
                ImageDraw.Draw(stroke).text((x, y), line, font=font, fill=255, anchor="ls",
                                            stroke_width=stroke_w, stroke_fill=255)
            if is_main:
                ImageDraw.Draw(main).text((x, y), line, font=font, fill=255, anchor="ls",
                                          stroke_width=stroke_w, stroke_fill=255)

        if rng.random() < 0.05:
            y = baseline + max(2, size // 8)
            ImageDraw.Draw(fill).line([(x0, y), (x0 + length, y)], fill=255, width=max(1, size // 18))
            ImageDraw.Draw(main).line([(x0, y), (x0 + length, y)], fill=255, width=max(1, size // 18))

        bg, fg = _colors(rng)
        img = _background(rng, height, width, bg)
        fill_a = np.asarray(fill, np.float32)[..., None] / 255
        if rng.random() < 0.1:
            # Тень обычно падает вниз: свет в реальных сценах чаще сверху.
            dy = int(rng.integers(1, max(2, size // 10) + 1)) * (1 if rng.random() < 0.8 else -1)
            dx = int(rng.integers(-size // 10, size // 10 + 1))
            shadow = np.roll(fill_a, (dy, dx), axis=(0, 1))
            img = img * (1 - shadow * 0.7)
        if stroke:
            stroke_a = np.asarray(stroke, np.float32)[..., None] / 255
            img = img * (1 - stroke_a) + (255 - fg) * stroke_a
        img = img * (1 - fill_a) + fg * fill_a

        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        img, main = self._warp(rng, img, main, size)
        img = self._crop(rng, img, main)
        return self._degrade(rng, img)

    @staticmethod
    def _warp(rng, img, mask, size):
        if rng.random() < 0.05:
            # Дуга «горкой», как на табличках с названием улицы.
            w, h = img.size
            amp = rng.uniform(0.1, 0.5) * size
            xs = np.linspace(-1, 1, w)
            dy = np.round(-amp * (1 - xs ** 2)).astype(int) + int(amp)
            rows = np.clip(np.arange(h)[:, None] - dy[None, :], 0, h - 1)
            cols = np.arange(w)[None, :]
            img = Image.fromarray(np.asarray(img)[rows, cols])
            mask = Image.fromarray(np.asarray(mask)[rows, cols])
        if rng.random() < 0.15:
            k = rng.uniform(-0.25, 0.25)
            w, h = img.size
            coeffs = (1, k, -k * h / 2, 0, 1, 0)
            img = img.transform(img.size, Image.AFFINE, coeffs, Image.BILINEAR)
            mask = mask.transform(mask.size, Image.AFFINE, coeffs, Image.BILINEAR)
        if rng.random() < 0.5:
            angle = rng.uniform(-2.5, 2.5)
            img = img.rotate(angle, Image.BILINEAR)
            mask = mask.rotate(angle, Image.BILINEAR)
        return img, mask

    @staticmethod
    def _crop(rng, img, mask):
        ys, xs = np.nonzero(np.asarray(mask) > 64)
        if len(ys) == 0:
            return img
        top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h = bottom - top
        # Поля сверху и снизу независимы: иначе их соотношение стало бы подсказкой.
        top -= int(h * rng.uniform(0.0, 0.35))
        bottom += int(h * rng.uniform(0.0, 0.35))
        left -= int(h * rng.uniform(0.0, 0.3))
        right += int(h * rng.uniform(0.0, 0.3))
        return img.crop((max(0, left), max(0, top), min(img.width, right), min(img.height, bottom)))

    @staticmethod
    def _degrade(rng, img):
        # Высота кропов в тесте от 10 до ~500 px, медиана 42: уменьшаем до похожего размера.
        target = int(np.exp(rng.uniform(np.log(14), np.log(110))))
        if target < img.height:
            img = img.resize((max(1, round(img.width * target / img.height)), target), Image.BILINEAR)
        if rng.random() < 0.25:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        if rng.random() < 0.2:
            a = np.asarray(img, np.float32) + rng.normal(0, rng.uniform(3, 15), (img.height, img.width, 1))
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if rng.random() < 0.6:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(rng.integers(25, 96)))
            img = Image.open(buf).convert("RGB")
        return img
