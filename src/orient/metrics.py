"""Метрики вероятностного прогноза и калибровка температурой."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def report(p, y) -> dict:
    """Основная метрика задания — score = 1 − Brier."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "score": 1 - brier(p, y),
        "brier": brier(p, y),
        "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        "ece": ece(p, y),
        "n": len(p),
    }


def reliability(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y})
    return df.groupby("bin").agg(p_mean=("p", "mean"), y_mean=("y", "mean"), n=("y", "size")).reset_index()


def ece(p, y, bins: int = 15) -> float:
    r = reliability(p, y, bins)
    return float(np.sum(r.n * np.abs(r.p_mean - r.y_mean)) / r.n.sum())


def fit_temperature(logits, y) -> float:
    """Температура, минимизирующая Brier: калибруем ровно под метрику задания."""
    logits, y = np.asarray(logits, float), np.asarray(y, float)
    res = minimize_scalar(lambda t: brier(sigmoid(logits / t), y), bounds=(0.05, 20), method="bounded")
    return float(res.x)
