"""Цикл обучения и предсказания."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from .metrics import report, sigmoid
from .models import tta_logits


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_epoch(ds, epoch: int) -> None:
    for d in ds.datasets if isinstance(ds, ConcatDataset) else [ds]:
        d.epoch = epoch


@torch.no_grad()
def predict(model, loader, dev, tta: bool = True, input_index: int = 0):
    """Логиты «перевёрнут» и метки (None для датасета без меток); с TTA логиты антисимметричные."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        x = batch[input_index] if isinstance(batch, (list, tuple)) else batch
        x = x.to(dev)
        z = tta_logits(model, x) if tta else model(x)
        logits.append(z.float().cpu().numpy()[:, 0])
        if isinstance(batch, (list, tuple)):
            labels.append(batch[-1].numpy()[:, 0])
    return np.concatenate(logits), (np.concatenate(labels) if labels else None)


def train(model, train_ds, val_loader, *, steps: int, batch_size: int, lr: float,
          weight_decay: float, warmup: int, eval_every: int, out_dir: Path, seed: int,
          teacher=None, alpha: float = 1.0, workers: int = 8) -> dict:
    """Обучение по числу шагов.

    Без учителя цель — метка датасета. С учителем цель — смесь
    alpha * p_учителя + (1 − alpha) * метка; p_учителя считается на холсте batch[0],
    студент учится на batch[1]. Валидация всегда на batch[0] своего val_loader.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = device()
    model.to(dev)
    if teacher is not None:
        teacher.to(dev).eval()
    student_input = 1 if teacher is not None else 0

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=workers,
                        persistent_workers=False, drop_last=True)
    log, step, epoch, t0 = [], 0, 0, time.time()
    best = {"score": -1.0}
    while step < steps:
        _set_epoch(train_ds, epoch)
        model.train()
        for batch in loader:
            x = batch[student_input].to(dev)
            y = batch[-1].to(dev)
            if teacher is not None:
                with torch.no_grad():
                    p_t = torch.sigmoid(tta_logits(teacher, batch[0].to(dev)))
                y = alpha * p_t + (1 - alpha) * y
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1

            if step % eval_every == 0 or step == steps:
                z, y_val = predict(model, val_loader, dev)
                m = report(sigmoid(z), y_val)
                m.update(step=step, loss=loss.item(), minutes=(time.time() - t0) / 60)
                log.append(m)
                print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}),
                      flush=True)
                torch.save(model.state_dict(), out_dir / "last.pt")
                if m["score"] > best["score"]:
                    best = m
                    torch.save(model.state_dict(), out_dir / "best.pt")
                (out_dir / "log.json").write_text(json.dumps(log, indent=1))
                model.train()
            if step >= steps:
                break
        epoch += 1
    return best
