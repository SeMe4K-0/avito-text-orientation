"""Этапы решения: кропы → учитель → студент → оценка и калибровка → submission.

Каждый этап сохраняет результат на диск и при повторном вызове не пересчитывается.
Запуск из solution.ipynb или из терминала: `python -m orient.pipeline <этап> [--key=value ...]`.
"""
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset

from . import infer, real
from .baselines import rapidocr_cls
from .dataset import FileSource, OrientDataset, SynthSource
from .metrics import fit_temperature, report, sigmoid
from .models import STUDENT, TEACHER, build, n_params
from .paths import DATA, ROOT, TEST_IMAGES
from .synth import SynthRenderer
from .text import TextSampler
from .train import device, predict, seed_everything, train

ARTIFACTS = ROOT / "artifacts"      # в репозитории: веса студента, ONNX, калибровка, метрики
CHECKPOINTS = ROOT / "checkpoints"  # вне репозитория: чекпойнт учителя весит ~110 МБ
CROPS = DATA / "crops"


@dataclass
class Config:
    seed: int = 42
    synth_per_epoch: int = 100_000  # синтетики в эпохе рядом с реальными кропами
    synth_val: int = 5_000
    val_during_train: int = 5_000   # подвыборка валидации для контроля во время обучения
    teacher_steps: int = 15_000
    teacher_batch: int = 64
    teacher_lr: float = 2e-4
    student_steps: int = 20_000
    student_batch: int = 128
    student_lr: float = 3e-3
    alpha: float = 0.5              # вес мягкой метки учителя в цели студента
    eval_every: int = 1_000
    workers: int = 10


# ---------- данные ----------

def renderer() -> SynthRenderer:
    corpora = DATA / "corpora"
    sampler = TextSampler(sorted(corpora.glob("rus_*-sentences.txt")),
                          sorted(corpora.glob("eng_*-sentences.txt")))
    return SynthRenderer(DATA / "fonts", sampler)


def prepare_crops(cfg: Config) -> pd.DataFrame:
    photos = sorted(p for p in (DATA / "photos").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos and not (CROPS / "crops.csv").exists():
        print("В data/photos нет фото: обучение и валидация только на синтетике")
        return pd.DataFrame(columns=["crop_id", "photo_id", "split", "w", "h"])
    return real.extract_crops(photos, CROPS, workers=cfg.workers)


def crop_paths(meta: pd.DataFrame, split: str) -> list[Path]:
    return [CROPS / "images" / f"{c}.png" for c in meta.loc[meta.split == split, "crop_id"]]


def val_sets(cfg: Config, rend, meta, heights) -> dict[str, OrientDataset]:
    sets = {"synth": OrientDataset(SynthSource(rend, cfg.synth_val, cfg.seed + 100), heights,
                                   cfg.seed + 101, rotation="fixed")}
    paths = crop_paths(meta, "val")
    if paths:
        sets["real"] = OrientDataset(FileSource(paths), heights, cfg.seed + 102, rotation="fixed")
    return sets


def control_loader(cfg: Config, sets: dict) -> DataLoader:
    """Валидация во время обучения: реальные кропы, если есть, иначе синтетика."""
    ds = sets.get("real", sets["synth"])
    idx = np.random.default_rng(cfg.seed).permutation(len(ds))[:cfg.val_during_train]
    return DataLoader(Subset(ds, sorted(idx)), batch_size=256, num_workers=cfg.workers)


def train_set(cfg: Config, rend, meta, heights, seed: int) -> ConcatDataset:
    parts = [OrientDataset(SynthSource(rend, cfg.synth_per_epoch, seed), heights, seed)]
    paths = crop_paths(meta, "train")
    if paths:
        parts.append(OrientDataset(FileSource(paths), heights, seed + 1, augment=True))
    return ConcatDataset(parts)


def load(spec: dict, path: Path) -> torch.nn.Module:
    model = build(spec["name"], pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


class Scaled(torch.nn.Module):
    """Модель с делением логита на температуру."""

    def __init__(self, model, t: float):
        super().__init__()
        self.model, self.t = model, t

    def forward(self, x):
        return self.model(x) / self.t


# ---------- обучение ----------

def train_teacher(cfg: Config) -> Path:
    out = CHECKPOINTS / "teacher"
    if (out / "best.pt").exists():
        return out / "best.pt"
    seed_everything(cfg.seed)
    rend, meta = renderer(), prepare_crops(cfg)
    h = (TEACHER["height"],)
    train(build(TEACHER["name"]), train_set(cfg, rend, meta, h, cfg.seed),
          control_loader(cfg, val_sets(cfg, rend, meta, h)),
          steps=cfg.teacher_steps, batch_size=cfg.teacher_batch, lr=cfg.teacher_lr, weight_decay=0.05,
          warmup=500, eval_every=cfg.eval_every, out_dir=out, seed=cfg.seed, workers=cfg.workers)
    ARTIFACTS.mkdir(exist_ok=True)
    shutil.copy(out / "log.json", ARTIFACTS / "teacher_log.json")
    return out / "best.pt"


def train_student(cfg: Config) -> Path:
    out = CHECKPOINTS / "student"
    if not (out / "best.pt").exists():
        seed_everything(cfg.seed)
        rend, meta = renderer(), prepare_crops(cfg)
        teacher = load(TEACHER, CHECKPOINTS / "teacher" / "best.pt")

        # Мягкие метки учителя должны быть откалиброваны, иначе студент унаследует его пере-/недоуверенность.
        dev = device()
        z, y = predict(teacher.to(dev), control_loader(cfg, val_sets(cfg, rend, meta, (TEACHER["height"],))), dev)
        t_teacher = fit_temperature(z, y)
        print(f"температура учителя: {t_teacher:.3f}")

        heights = (TEACHER["height"], STUDENT["height"])
        train(build(STUDENT["name"]), train_set(cfg, rend, meta, heights, cfg.seed + 10),
              control_loader(cfg, val_sets(cfg, rend, meta, (STUDENT["height"],))),
              steps=cfg.student_steps, batch_size=cfg.student_batch, lr=cfg.student_lr, weight_decay=0.01,
              warmup=500, eval_every=cfg.eval_every, out_dir=out, seed=cfg.seed,
              teacher=Scaled(teacher, t_teacher), alpha=cfg.alpha, workers=cfg.workers)
    ARTIFACTS.mkdir(exist_ok=True)
    shutil.copy(out / "best.pt", ARTIFACTS / "student.pt")
    shutil.copy(out / "log.json", ARTIFACTS / "student_log.json")
    return ARTIFACTS / "student.pt"


# ---------- оценка и калибровка ----------

def image_sizes(paths: list[Path]) -> np.ndarray:
    return np.array([Image.open(p).size for p in paths])


def test_weights(val_wh: np.ndarray, test_wh: np.ndarray) -> np.ndarray:
    """Веса валидации, выравнивающие её распределение (высота × пропорции) с тестом."""
    def cell(wh):
        h, ar = wh[:, 1], wh[:, 0] / wh[:, 1]
        return np.digitize(h, [16, 24, 36, 54, 80, 120]) * 10 + np.digitize(ar, [2, 3, 4.5, 6.5, 10])

    v, t = pd.Series(cell(val_wh)), pd.Series(cell(test_wh))
    share_t, share_v = t.value_counts(normalize=True), v.value_counts(normalize=True)
    w = share_t.reindex(v).fillna(0).to_numpy() / share_v.reindex(v).to_numpy()
    return w / w.mean()


def hash_fold(photo_id: str) -> int:
    """Фолд для кросс-фиттинга температуры: кропы одного фото всегда в одном фолде."""
    return int(hashlib.md5(photo_id.encode()).hexdigest()[-1], 16) % 2


def oof_calibrated(z: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Температура подбирается на одном фолде и применяется к другому — честная оценка калибровки."""
    p = np.empty_like(z, dtype=float)
    for f in np.unique(folds):
        t = fit_temperature(z[folds != f], y[folds != f])
        p[folds == f] = sigmoid(z[folds == f] / t)
    return p


def evaluate(cfg: Config) -> pd.DataFrame:
    """Все модели на полной валидации; сохраняет предсказания, метрики и температуру студента."""
    rend, meta = renderer(), prepare_crops(cfg)
    dev = device()
    teacher_path = CHECKPOINTS / "teacher" / "best.pt"
    teacher = load(TEACHER, teacher_path).to(dev) if teacher_path.exists() else None
    student = load(STUDENT, ARTIFACTS / "student.pt").to(dev)
    test_wh = image_sizes(infer.test_paths(TEST_IMAGES))

    frames = []
    for name, ds in val_sets(cfg, rend, meta, (TEACHER["height"], STUDENT["height"])).items():
        loader = DataLoader(ds, batch_size=256, num_workers=cfg.workers)
        images, labels = zip(*(ds.image(i) for i in range(len(ds))))
        df = pd.DataFrame({"set": name, "y": labels, "w": [im.width for im in images],
                           "h": [im.height for im in images]})
        if name == "real":
            ids = meta.loc[meta.split == "val"]
            df["crop_id"], df["photo_id"] = ids.crop_id.to_numpy(), ids.photo_id.to_numpy()
            df["fold"] = df.photo_id.map(hash_fold)
        else:
            df["fold"] = np.arange(len(df)) % 2
        df["z_student"], _ = predict(student, loader, dev, input_index=1)
        df["z_student_notta"], _ = predict(student, loader, dev, tta=False, input_index=1)
        if teacher is not None:
            df["z_teacher"], _ = predict(teacher, loader, dev, input_index=0)
        df["p_rapidocr"] = rapidocr_cls(list(images))
        frames.append(df)
    preds = pd.concat(frames, ignore_index=True)

    rows = []
    for name, df in preds.groupby("set"):
        y = df.y.to_numpy()
        weights = test_weights(df[["w", "h"]].to_numpy(), test_wh) if name == "real" else None
        cands = {"константа 0.5": np.full(len(df), 0.5), "RapidOCR cls": df.p_rapidocr.to_numpy()}
        for col, label in [("z_teacher", "учитель"), ("z_student_notta", "студент без TTA"),
                           ("z_student", "студент")]:
            if col in df:
                z = df[col].to_numpy()
                cands[label] = sigmoid(z)
                cands[label + " + T"] = oof_calibrated(z, y, df.fold.to_numpy())
        for label, p in cands.items():
            r = {"set": name, "model": label, **report(p, y)}
            if weights is not None:
                r["score_test_weighted"] = 1 - float(np.average((p - y) ** 2, weights=weights))
            rows.append(r)
    metrics = pd.DataFrame(rows)

    ARTIFACTS.mkdir(exist_ok=True)
    main = preds[preds.set == ("real" if "real" in preds.set.values else "synth")]
    calib = {"temperature": fit_temperature(main.z_student.to_numpy(), main.y.to_numpy()), "tta": True,
             "fitted_on": str(main.set.iloc[0]), "n": int(len(main))}
    (ARTIFACTS / "calibration.json").write_text(json.dumps(calib, indent=1))
    preds.to_csv(ARTIFACTS / "val_predictions.csv", index=False)
    metrics.to_csv(ARTIFACTS / "metrics.csv", index=False)
    return metrics


# ---------- submission и скорость ----------

def make_submission(path: Path = ROOT / "submission.csv") -> pd.DataFrame:
    calib = json.loads((ARTIFACTS / "calibration.json").read_text())
    model = load(STUDENT, ARTIFACTS / "student.pt")
    df = infer.predict_test(model, STUDENT["height"], infer.test_paths(TEST_IMAGES),
                            calib["temperature"], tta=calib["tta"])
    infer.write_submission(df, path)
    return df


def benchmark() -> pd.DataFrame:
    calib = json.loads((ARTIFACTS / "calibration.json").read_text())
    model = load(STUDENT, ARTIFACTS / "student.pt")
    h = STUDENT["height"]
    paths = infer.test_paths(TEST_IMAGES)
    rows = []
    for tta in (False, True):
        onnx_path = ARTIFACTS / f"student{'_tta' if tta else ''}.onnx"
        infer.export_onnx(infer.Deployed(model, calib["temperature"], tta), h, h * 10, onnx_path)
        for threads, batch in [(1, 1), (1, 32), (4, 32)]:
            rows.append({"tta": tta, "params_m": n_params(model) / 1e6,
                         **infer.benchmark_onnx(onnx_path, paths, h, threads, batch)})
    return pd.DataFrame(rows)


STAGES = {"crops": prepare_crops, "teacher": train_teacher, "student": train_student,
          "evaluate": evaluate}


def main(argv: list[str]) -> None:
    stage, overrides = argv[0], dict(a.lstrip("-").split("=", 1) for a in argv[1:])
    types = {f.name: f.type for f in fields(Config)}
    cfg = Config(**{k: (None if v == "None" else int(v) if "int" in str(types[k]) else
                        float(v) if types[k] is float else v) for k, v in overrides.items()})
    print(json.dumps(asdict(cfg)), flush=True)
    if stage == "submission":
        make_submission()
    elif stage == "benchmark":
        print(benchmark().to_string())
    else:
        print(STAGES[stage](cfg))


if __name__ == "__main__":
    main(sys.argv[1:])
