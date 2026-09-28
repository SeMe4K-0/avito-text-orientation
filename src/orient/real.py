"""Нарезка реальных кропов из фото детектором PP-OCRv4 (RapidOCR).

Так получаются кропы той же геометрии, что и в тесте: детектор PaddleOCR-типа,
перспективная вырезка бокса, поля от unclip.
"""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

MIN_H, MIN_W = 8, 16
VAL_SHARE = 0.1

_engine = None


def _init_engine() -> None:
    global _engine
    from rapidocr_onnxruntime import RapidOCR

    # По 2 потока на процесс: параллелим по фото, а не внутри одной сети.
    _engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)


def split_of(photo_id: str) -> str:
    """Train/val по хэшу имени фото: все кропы одного фото попадают в одну часть."""
    h = int(hashlib.md5(photo_id.encode()).hexdigest()[:8], 16)
    return "val" if h % 1000 < VAL_SHARE * 1000 else "train"


def detect_crops(data: bytes) -> list[np.ndarray]:
    """Горизонтальные кропы текста (RGB) из одного фото."""
    e = _engine
    img = e.load_img(data)
    img, _, _ = e.preprocess(img)
    img, _ = e.maybe_add_letterbox(img, {})
    boxes, _ = e.auto_text_det(img)
    if boxes is None:
        return []
    crops = []
    for box in boxes:
        w = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3]))
        h = max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2]))
        # Высокие боксы RapidOCR поворачивает на 90°, и ориентация таких кропов не определена.
        if h > w or h < MIN_H or w < MIN_W:
            continue
        crop = e.get_crop_img_list(img, [box])[0]
        crops.append(np.ascontiguousarray(crop[..., ::-1]))
    return crops


def _process(args):
    photo, out_dir = args
    photo_id = photo.stem
    try:
        crops = detect_crops(photo.read_bytes())
    except Exception:
        return []
    rows = []
    for k, crop in enumerate(crops):
        crop_id = f"{photo_id}_{k}"
        Image.fromarray(crop).save(out_dir / f"{crop_id}.png")
        rows.append({"crop_id": crop_id, "photo_id": photo_id, "split": split_of(photo_id),
                     "w": crop.shape[1], "h": crop.shape[0]})
    return rows


def extract_crops(photos: list[Path], out_dir: Path, workers: int = 6) -> pd.DataFrame:
    """Режет фото на кропы, сохраняет их PNG и таблицу `crops.csv`."""
    meta_path = out_dir / "crops.csv"
    if meta_path.exists():
        # Имена фото вида «00012» без dtype прочитались бы как числа.
        return pd.read_csv(meta_path, dtype={"crop_id": str, "photo_id": str})
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(p, img_dir) for p in sorted(photos)]
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init_engine) as pool:
        for r in tqdm(pool.map(_process, tasks, chunksize=8), total=len(tasks), desc="photos"):
            rows += r
    meta = pd.DataFrame(rows, columns=["crop_id", "photo_id", "split", "w", "h"])
    meta.to_csv(meta_path, index=False)
    return meta
