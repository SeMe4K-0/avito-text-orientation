# Данные

В репозитории лежат только открытые данные, которые можно распространять.

| Папка | Источник | Лицензия |
|---|---|---|
| `fonts/` | [Google Fonts](https://github.com/google/fonts), 71 семейство с кириллицей | SIL OFL 1.1 / Apache 2.0 / UFL — файл лицензии лежит в папке каждого семейства |
| `corpora/` | [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download): предложения из русской и английской Википедии (D. Goldhahn, T. Eckart, U. Quasthoff. Building Large Monolingual Dictionaries at the Leipzig Corpora Collection. LREC 2012) | тексты Википедии — CC BY-SA |

## Рукописный датасет

Нужен только для обучения; для воспроизведения `submission.csv` не требуется.

1. Скачайте [Cyrillic Handwriting Dataset](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset) (нужен аккаунт Kaggle).
2. Распакуйте архив в `data/handwriting/` так, чтобы получились `train/`, `test/`, `train.tsv`, `test.tsv`.

Из обучающей части берётся подвыборка 15 000 строк, из тестовой — 1 500 для валидации. Состав подвыборок фиксирован сидом.

## Веса ImageNet

`download.download_pretrained()` кладёт их в кэш HuggingFace (`~/.cache/huggingface`). Нужны только для обучения с нуля.
