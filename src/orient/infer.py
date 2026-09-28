"""Предсказание на тесте, экспорт в ONNX и замер скорости."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import FolderDataset
from .metrics import sigmoid
from .preprocess import to_canvas
from .train import predict


class Deployed(torch.nn.Module):
    """Студент в том виде, в каком он работает в проде: холст → p_180 (TTA и температура внутри)."""

    def __init__(self, model: torch.nn.Module, temperature: float, tta: bool):
        super().__init__()
        self.model, self.t, self.tta = model, temperature, tta

    def forward(self, x):
        z = self.model(x)
        if self.tta:
            z = (z - self.model(torch.flip(x, dims=(2, 3)))) / 2
        return torch.sigmoid(z / self.t)


def test_paths(test_dir: Path) -> list[Path]:
    return sorted(Path(test_dir).glob("*.png"))


def predict_test(model, height: int, paths: list[Path], temperature: float, tta: bool = True,
                 batch_size: int = 256, workers: int = 4) -> pd.DataFrame:
    """Инференс на CPU: так результат не зависит от GPU/MPS и воспроизводится побитово."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        loader = DataLoader(FolderDataset(paths, height), batch_size=batch_size, num_workers=workers)
        z, _ = predict(model.cpu(), loader, torch.device("cpu"), tta=tta)
    finally:
        torch.use_deterministic_algorithms(prev)
    return pd.DataFrame({"image_id": [p.stem for p in paths], "p_180": sigmoid(z / temperature)})


def write_submission(df: pd.DataFrame, path: Path) -> None:
    assert df["p_180"].between(0, 1).all() and df["p_180"].notna().all()
    df.to_csv(path, index=False, float_format="%.6f")


def export_onnx(deployed: Deployed, height: int, width: int, path: Path) -> None:
    deployed.eval().cpu()
    dummy = torch.zeros(1, 1, height, width)
    torch.onnx.export(deployed, dummy, str(path), input_names=["canvas"], output_names=["p_180"],
                      dynamic_axes={"canvas": {0: "batch"}, "p_180": {0: "batch"}},
                      opset_version=17, dynamo=False)


def benchmark_onnx(path: Path, paths: list[Path], height: int, threads: int = 1,
                   batch_size: int = 1, n: int = 2000) -> dict:
    """Время на кроп в ONNX Runtime на CPU: отдельно подготовка холста и сеть."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    imgs = [Image.open(p).convert("RGB") for p in paths[:n]]

    t = time.perf_counter()
    canvases = np.stack([to_canvas(im, height) for im in imgs])
    prep = (time.perf_counter() - t) / len(imgs)

    sess.run(None, {"canvas": canvases[:batch_size]})
    t = time.perf_counter()
    for i in range(0, len(canvases), batch_size):
        sess.run(None, {"canvas": canvases[i:i + batch_size]})
    net = (time.perf_counter() - t) / len(canvases)
    return {"threads": threads, "batch": batch_size, "prep_ms": prep * 1e3, "net_ms": net * 1e3,
            "crops_per_s": 1 / (prep + net), "onnx_mb": Path(path).stat().st_size / 1e6}
