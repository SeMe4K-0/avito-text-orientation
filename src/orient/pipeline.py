"""Этапы решения: данные → учитель → студент → валидация → submission.

Каждый этап сохраняет результат на диск и при повторном вызове не пересчитывается.
Запуск из solution.ipynb или из терминала:
    python -m orient.pipeline teacher student evaluate submission
"""
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

from . import infer
from .baselines import rapidocr_cls
from .dataset import FileSource, OrientDataset, SynthSource
from .metrics import fit_temperature, report, sigmoid
from .models import STUDENT, TEACHER, build, n_params
from .paths import DATA, ROOT, TEST_IMAGES
from .synth import SynthRenderer
from .text import TextSampler
from .train import device, predict, seed_everything, train

ARTIFACTS = ROOT / "artifacts"      # в репозитории: веса, метрики, предсказания валидации
CHECKPOINTS = ROOT / "checkpoints"  # рабочие чекпойнты обучения, вне репозитория

# Итоговая модель — учитель с антисимметричным TTA. Температура 1.6 подобрана
# по трём результатам платформы (см. README, «Калибровка по лидерборду»), а не по валидации:
# на валидации модель выглядит почти идеально откалиброванной, на тесте — переуверенной.
SUBMISSION_TEMPERATURE = 1.6


@dataclass
class Config:
    seed: int = 42
    run: str = "v1"                  # имя прогона: чекпойнты в checkpoints/{teacher,student}_{run}
    synth_per_epoch: int = 100_000   # синтетических примеров в эпохе
    synth_val: int = 3_000
    handwriting_train: int = 15_000  # рукописи в тесте немного, берём ограниченную подвыборку
    handwriting_val: int = 1_500
    val_during_train: int = 1_500    # подвыборка валидации для контроля во время обучения
    teacher_steps: int = 1_200
    teacher_batch: int = 64
    teacher_lr: float = 2e-4
    student_steps: int = 2_000
    student_batch: int = 128
    student_lr: float = 3e-3
    alpha: float = 0.5               # вес мягкой метки учителя в цели студента
    eval_every: int = 400
    workers: int = 10


# ---------- данные ----------

def renderer() -> SynthRenderer:
    corpora = DATA / "corpora"
    sampler = TextSampler(sorted(corpora.glob("rus_*-sentences.txt")),
                          sorted(corpora.glob("eng_*-sentences.txt")))
    return SynthRenderer(DATA / "fonts", sampler)


def handwriting_paths(split: str, limit: int, seed: int) -> list[Path]:
    """Cyrillic Handwriting Dataset: папки train/ и test/ с ровными рукописными строками."""
    paths = sorted((DATA / "handwriting" / split).glob("*.png"))
    if len(paths) > limit:
        idx = np.random.default_rng(seed).choice(len(paths), limit, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    return paths


def val_sets(cfg: Config, rend, heights) -> dict[str, OrientDataset]:
    """rotation='fixed' — поворот зависит только от индекса, поэтому валидация неизменна."""
    sets = {"synth": OrientDataset(SynthSource(rend, cfg.synth_val, cfg.seed + 100), heights,
                                   cfg.seed + 101, rotation="fixed")}
    hw = handwriting_paths("test", cfg.handwriting_val, cfg.seed)
    if hw:
        sets["handwriting"] = OrientDataset(FileSource(hw), heights, cfg.seed + 103, rotation="fixed")
    return sets


def control_loader(cfg: Config, sets: dict) -> DataLoader:
    """Контроль во время обучения — рукопись: единственные реальные фотографии в валидации."""
    ds = sets.get("handwriting", sets["synth"])
    idx = np.random.default_rng(cfg.seed).permutation(len(ds))[:cfg.val_during_train]
    return DataLoader(Subset(ds, sorted(idx)), batch_size=256, num_workers=cfg.workers)


def train_set(cfg: Config, rend, heights, seed: int) -> ConcatDataset:
    parts = [OrientDataset(SynthSource(rend, cfg.synth_per_epoch, seed), heights, seed)]
    hw = handwriting_paths("train", cfg.handwriting_train, cfg.seed)
    if hw:
        parts.append(OrientDataset(FileSource(hw), heights, seed + 2, augment=True))
    return ConcatDataset(parts)


# Чекпойнт учителя (111 МБ) не помещается в лимит GitHub на один файл,
# поэтому в репозитории он лежит частями teacher.pt.part00, part01, ...
PART_SIZE = 64 << 20


def split_file(path: Path, part_size: int = PART_SIZE) -> list[Path]:
    data = path.read_bytes()
    parts = []
    for i in range(0, len(data), part_size):
        part = path.with_suffix(path.suffix + f".part{i // part_size:02d}")
        part.write_bytes(data[i:i + part_size])
        parts.append(part)
    return parts


def join_file(path: Path) -> Path:
    """Собирает файл из частей, если его самого нет."""
    if path.exists():
        return path
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not parts:
        raise FileNotFoundError(path)
    path.write_bytes(b"".join(p.read_bytes() for p in parts))
    return path


def load(spec: dict, path: Path) -> torch.nn.Module:
    model = build(spec["name"], pretrained=False)
    model.load_state_dict(torch.load(join_file(path), map_location="cpu"))
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
    out = CHECKPOINTS / f"teacher_{cfg.run}"
    if not (out / "best.pt").exists():
        seed_everything(cfg.seed)
        rend = renderer()
        h = (TEACHER["height"],)
        train(build(TEACHER["name"]), train_set(cfg, rend, h, cfg.seed),
              control_loader(cfg, val_sets(cfg, rend, h)),
              steps=cfg.teacher_steps, batch_size=cfg.teacher_batch, lr=cfg.teacher_lr,
              weight_decay=0.05, warmup=500, eval_every=cfg.eval_every, out_dir=out,
              seed=cfg.seed, workers=cfg.workers)
    ARTIFACTS.mkdir(exist_ok=True)
    shutil.copy(out / "best.pt", ARTIFACTS / "teacher.pt")
    split_file(ARTIFACTS / "teacher.pt")
    shutil.copy(out / "log.json", ARTIFACTS / "teacher_log.json")
    return ARTIFACTS / "teacher.pt"


def train_student(cfg: Config) -> Path:
    out = CHECKPOINTS / f"student_{cfg.run}"
    if not (out / "best.pt").exists():
        seed_everything(cfg.seed)
        rend = renderer()
        teacher = load(TEACHER, CHECKPOINTS / f"teacher_{cfg.run}" / "best.pt")

        # Мягкие метки учителя калибруем: иначе студент унаследует его пере- или недоуверенность.
        dev = device()
        z, y = predict(teacher.to(dev), control_loader(cfg, val_sets(cfg, rend, (TEACHER["height"],))), dev)
        t_teacher = fit_temperature(z, y)
        print(f"температура учителя: {t_teacher:.3f}")

        # Два холста в батче: 48 px для учителя, 32 px для студента.
        heights = (TEACHER["height"], STUDENT["height"])
        train(build(STUDENT["name"]), train_set(cfg, rend, heights, cfg.seed + 10),
              control_loader(cfg, val_sets(cfg, rend, (STUDENT["height"],))),
              steps=cfg.student_steps, batch_size=cfg.student_batch, lr=cfg.student_lr,
              weight_decay=0.01, warmup=500, eval_every=cfg.eval_every, out_dir=out,
              seed=cfg.seed, teacher=Scaled(teacher, t_teacher), alpha=cfg.alpha, workers=cfg.workers)
    ARTIFACTS.mkdir(exist_ok=True)
    shutil.copy(out / "best.pt", ARTIFACTS / "student.pt")
    shutil.copy(out / "log.json", ARTIFACTS / "student_log.json")
    return ARTIFACTS / "student.pt"


# ---------- валидация ----------

def oof_calibrated(z: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Температура подбирается на одном фолде и применяется к другому — честная оценка калибровки."""
    p = np.empty_like(z, dtype=float)
    for f in np.unique(folds):
        t = fit_temperature(z[folds != f], y[folds != f])
        p[folds == f] = sigmoid(z[folds == f] / t)
    return p


def evaluate(cfg: Config) -> pd.DataFrame:
    """Учитель, студент и бейзлайны на полной валидации; сохраняет предсказания и метрики."""
    rend = renderer()
    dev = device()
    teacher = load(TEACHER, ARTIFACTS / "teacher.pt").to(dev)
    student = load(STUDENT, ARTIFACTS / "student.pt").to(dev)

    frames = []
    for name, ds in val_sets(cfg, rend, (TEACHER["height"], STUDENT["height"])).items():
        loader = DataLoader(ds, batch_size=256, num_workers=cfg.workers)
        images, labels = zip(*(ds.image(i) for i in range(len(ds))))
        df = pd.DataFrame({"set": name, "y": labels, "w": [im.width for im in images],
                           "h": [im.height for im in images], "fold": np.arange(len(labels)) % 2})
        df["z_teacher"], _ = predict(teacher, loader, dev, input_index=0)
        df["z_student"], _ = predict(student, loader, dev, input_index=1)
        df["z_student_notta"], _ = predict(student, loader, dev, tta=False, input_index=1)
        df["p_rapidocr"] = rapidocr_cls(list(images))
        frames.append(df)
    preds = pd.concat(frames, ignore_index=True)

    rows = []
    for name, df in preds.groupby("set"):
        y = df.y.to_numpy()
        cands = {"константа 0.5": np.full(len(df), 0.5), "RapidOCR cls": df.p_rapidocr.to_numpy()}
        for col, label in [("z_teacher", "учитель"), ("z_student_notta", "студент без TTA"),
                           ("z_student", "студент")]:
            z = df[col].to_numpy()
            cands[label] = sigmoid(z)
            cands[label + " + T"] = oof_calibrated(z, y, df.fold.to_numpy())
        cands[f"учитель, T={SUBMISSION_TEMPERATURE} (отправлено)"] = \
            sigmoid(df.z_teacher.to_numpy() / SUBMISSION_TEMPERATURE)
        rows += [{"set": name, "model": label, **report(p, y)} for label, p in cands.items()]

    ARTIFACTS.mkdir(exist_ok=True)
    preds.to_csv(ARTIFACTS / "val_predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ARTIFACTS / "metrics.csv", index=False)
    return metrics


# ---------- submission и скорость ----------

def make_submission(path: Path = ROOT / "submission.csv", temperature: float = SUBMISSION_TEMPERATURE,
                    workers: int = 4) -> pd.DataFrame:
    """Итоговые предсказания: учитель с антисимметричным TTA, вероятности с температурой."""
    model = load(TEACHER, ARTIFACTS / "teacher.pt")
    df = infer.predict_test(model, TEACHER["height"], infer.test_paths(TEST_IMAGES),
                            temperature, tta=True, batch_size=64, workers=workers)
    infer.write_submission(df, path)
    return df


def benchmark() -> pd.DataFrame:
    """Скорость студента в ONNX на CPU: он остаётся кандидатом для прода."""
    model = load(STUDENT, ARTIFACTS / "student.pt")
    h = STUDENT["height"]
    paths = infer.test_paths(TEST_IMAGES)
    rows = []
    for tta in (False, True):
        onnx_path = ARTIFACTS / f"student{'_tta' if tta else ''}.onnx"
        infer.export_onnx(infer.Deployed(model, 1.0, tta), h, h * 10, onnx_path)
        for threads, batch in [(1, 1), (1, 32), (4, 32)]:
            rows.append({"tta": tta, "params_m": n_params(model) / 1e6,
                         **infer.benchmark_onnx(onnx_path, paths, h, threads, batch)})
    return pd.DataFrame(rows)


STAGES = {"teacher": train_teacher, "student": train_student, "evaluate": evaluate}


def parse_config(overrides: dict[str, str]) -> Config:
    types = {f.name: str(f.type) for f in fields(Config)}

    def cast(key: str, value: str):
        kind = types[key]
        if "int" in kind:
            return int(value)
        if "float" in kind:
            return float(value)
        return value

    return Config(**{k: cast(k, v) for k, v in overrides.items()})


def main(argv: list[str]) -> None:
    stages = [a for a in argv if not a.startswith("-")]
    cfg = parse_config(dict(a.lstrip("-").split("=", 1) for a in argv if a.startswith("-")))
    print(json.dumps(asdict(cfg)), flush=True)
    for stage in stages:
        print(f"=== {stage}", flush=True)
        if stage == "submission":
            print(make_submission().head().to_string(), flush=True)
        elif stage == "benchmark":
            print(benchmark().round(3).to_string(), flush=True)
        else:
            print(STAGES[stage](cfg), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
