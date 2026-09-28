"""Готовый классификатор 0°/180° из PaddleOCR (ch_ppocr_mobile_v2.0_cls, через RapidOCR) — нулевой бейзлайн."""
import numpy as np
from PIL import Image


def rapidocr_cls(images: list[Image.Image], chunk: int = 512) -> np.ndarray:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    out = []
    for i in range(0, len(images), chunk):
        # RapidOCR ждёт BGR, как cv2.
        bgr = [np.ascontiguousarray(np.asarray(im.convert("RGB"))[..., ::-1]) for im in images[i:i + chunk]]
        _, res, _ = engine.text_cls(bgr)
        out += [score if label == "180" else 1 - score for label, score in res]
    return np.asarray(out, dtype=float)
