"""Строки для синтетики: фрагменты предложений из корпусов и шаблоны, типичные для объявлений."""
from pathlib import Path

import numpy as np

# Доли типов строк подобраны по тесту: в основном короткие русские фразы,
# заметная доля цифр (телефоны, цены, артикулы) и латиницы (бренды, модели).
KINDS = ["ru", "en", "number", "model"]
KIND_P = [0.5, 0.15, 0.2, 0.15]

# Длина фрагмента в словах: в тесте много одиночных слов и коротких фраз.
N_WORDS = np.arange(1, 11)
N_WORDS_P = np.array([25, 20, 15, 10, 8, 7, 5, 4, 3, 3], dtype=float)
N_WORDS_P /= N_WORDS_P.sum()

MODEL_SUFFIX = ["Pro", "Max", "Plus", "Lite", "Mini", "Ultra", "GB", "Gb", "mm", "W", "V", ""]
PRICE_UNITS = [" ₽", "₽", " руб.", " р.", " руб", ""]
LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"


def load_sentences(path: Path) -> list[str]:
    """Файл Leipzig: `id<TAB>предложение` в каждой строке."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t", 1)[-1] for line in f if "\t" in line]


class TextSampler:
    def __init__(self, ru_files: list[Path], en_files: list[Path]):
        self.ru = [s for p in ru_files for s in load_sentences(p)]
        self.en = [s for p in en_files for s in load_sentences(p)]
        self.en_words = [w for s in self.en[:20000] for w in s.split() if w.isalpha()]

    def sample(self, rng: np.random.Generator) -> str:
        kind = rng.choice(KINDS, p=KIND_P)
        if kind == "ru":
            text = self._fragment(rng, self.ru)
        elif kind == "en":
            text = self._fragment(rng, self.en)
        elif kind == "number":
            text = self._number(rng)
        else:
            text = self._model(rng)
        return self._case(rng, text)

    @staticmethod
    def _fragment(rng, sentences: list[str]) -> str:
        words = sentences[rng.integers(len(sentences))].split()
        n = min(len(words), rng.choice(N_WORDS, p=N_WORDS_P))
        start = rng.integers(len(words) - n + 1)
        return " ".join(words[start:start + n])

    @staticmethod
    def _case(rng, text: str) -> str:
        r = rng.random()
        if r < 0.25:
            return text.upper()
        if r < 0.35:
            return text.capitalize()
        return text

    @staticmethod
    def _digits(rng, n: int) -> str:
        return "".join(map(str, rng.integers(0, 10, n)))

    def _number(self, rng) -> str:
        d = lambda n: self._digits(rng, n)  # noqa: E731
        templates = [
            lambda: f"+7 (9{d(2)}) {d(3)}-{d(2)}-{d(2)}",
            lambda: f"8 9{d(2)} {d(3)} {d(2)} {d(2)}",
            lambda: f"8{d(10)}",
            lambda: f"{rng.integers(1, 999999):,}".replace(",", " ") + rng.choice(PRICE_UNITS),
            lambda: f"от {rng.integers(100, 99999)}" + rng.choice(PRICE_UNITS),
            lambda: f"{rng.integers(1, 29):02d}.{rng.integers(1, 13):02d}.{rng.integers(1990, 2027)}",
            lambda: f"с {rng.integers(7, 12)}:00 до {rng.integers(17, 24)}:00",
            lambda: "".join(rng.choice(list(LETTERS), rng.integers(1, 4))) + "-" + d(rng.integers(3, 8)),
            lambda: d(rng.integers(6, 20)),
            lambda: f"{rng.integers(1, 60)}-{rng.integers(1, 60)}",
            lambda: f"{rng.integers(1, 1000)} {rng.choice(['мл', 'л', 'кг', 'г', 'см', 'мм', 'шт', 'м²'])}",
        ]
        return templates[rng.integers(len(templates))]()

    def _model(self, rng) -> str:
        brand = self.en_words[rng.integers(len(self.en_words))].capitalize()
        parts = [brand, f"{rng.integers(1, 999)}", rng.choice(MODEL_SUFFIX)]
        if rng.random() < 0.3:
            parts.insert(1, "".join(rng.choice(list(LETTERS), rng.integers(1, 3))))
        return " ".join(p for p in parts if p)
