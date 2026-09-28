"""Картинки и графики для notebook."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .metrics import reliability


def contact_sheet(images: list[Image.Image], captions: list[str] | None = None, height: int = 36,
                  width: int = 1200, cols: int = 2) -> Image.Image:
    """Кропы в сетку, все приведены к одной высоте."""
    cell_w, cap_h = width // cols, 14 if captions else 0
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (width, rows * (height + cap_h + 6)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(images):
        x, y = (i % cols) * cell_w, (i // cols) * (height + cap_h + 6)
        w = min(cell_w - 10, max(1, round(im.width * height / im.height)))
        sheet.paste(im.convert("RGB").resize((w, height)), (x, y + cap_h))
        if captions:
            draw.text((x, y), captions[i], fill=(200, 0, 0))
    return sheet


def plot_logs(logs: dict[str, Path], ax=None):
    """Score на контрольной валидации по шагам обучения."""
    ax = ax or plt.gca()
    for name, path in logs.items():
        if Path(path).exists():
            log = json.loads(Path(path).read_text())
            ax.plot([r["step"] for r in log], [r["score"] for r in log], marker="o", label=name)
    ax.set_xlabel("шаг")
    ax.set_ylabel("1 − Brier")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_reliability(preds: dict[str, np.ndarray], y: np.ndarray, bins: int = 10, ax=None):
    """Диаграмма надёжности: у откалиброванной модели точки лежат на диагонали."""
    ax = ax or plt.gca()
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, p in preds.items():
        r = reliability(p, y, bins)
        ax.plot(r.p_mean, r.y_mean, marker="o", label=name)
    ax.set_xlabel("предсказанная p_180")
    ax.set_ylabel("доля перевёрнутых")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax
