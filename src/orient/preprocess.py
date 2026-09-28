"""Приведение кропа к входу сети: серый холст фиксированного размера."""
import numpy as np
from PIL import Image

# Длинные строки сжимаются по горизонтали до MAX_ASPECT: признаки ориентации
# (выносные элементы, базовая линия) вертикальные и при этом сохраняются.
MAX_ASPECT = 10


def to_canvas(img: Image.Image, height: int, max_aspect: float = MAX_ASPECT) -> np.ndarray:
    """Возвращает массив float32 [1, height, height * max_aspect] в диапазоне [-1, 1].

    Кроп ставится по центру, поэтому поворот холста на 180° с точностью до пикселя
    равен холсту от повёрнутого кропа — на этом держится антисимметричный TTA.
    """
    width = int(height * max_aspect)
    g = img.convert("L")
    w = min(width, max(1, round(g.width * height / g.height)))
    g = np.asarray(g.resize((w, height), Image.BILINEAR), dtype=np.float32)

    # Фон — медиана краёв кропа, чтобы граница вставки не выглядела как отдельный объект.
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    canvas = np.full((height, width), np.median(border), dtype=np.float32)
    left = (width - w) // 2
    canvas[:, left:left + w] = g
    return (canvas / 127.5 - 1.0)[None]
