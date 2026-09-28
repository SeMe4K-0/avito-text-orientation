"""Скачивание шрифтов, текстовых корпусов и предобученных весов.

Источники открытые и не требуют токенов. Функции можно перезапускать:
уже скачанное пропускается, оборванные загрузки докачиваются.
"""
import tarfile
import time
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm.auto import tqdm

from .paths import DATA

HEADERS = {"User-Agent": "avito-text-orientation/1.0 (https://github.com/SeMe4K-0/avito-text-orientation)"}

LEIPZIG_URL = "https://downloads.wortschatz-leipzig.de/corpora/{}.tar.gz"
# Только Википедия: её тексты под CC BY-SA, их можно распространять вместе с репозиторием.
CORPORA = ["rus_wikipedia_2021_100K", "eng_wikipedia_2016_100K"]

GF_TREE_URL = "https://api.github.com/repos/google/fonts/git/trees/main?recursive=1"
GF_RAW_URL = "https://raw.githubusercontent.com/google/fonts/main/{}"
FONT_LICENSES = ("OFL.txt", "LICENSE.txt", "UFL.txt")

# Папки семейств в репозитории google/fonts. Подобраны под стили из теста:
# гротески интерфейсов и баннеров, антиквы, моноширинные для артикулов,
# узкие для вывесок, рукописные и декоративные.
FONT_FAMILIES = [
    # гротески
    "ptsans", "ptsansnarrow", "ptsanscaption", "roboto", "robotocondensed", "opensans",
    "notosans", "montserrat", "rubik", "firasans", "firasanscondensed", "ibmplexsans",
    "exo2", "jura", "play", "scada", "arsenal", "manrope", "inter", "raleway", "nunito",
    "istokweb", "didactgothic", "commissioner", "golostext", "onest", "ubuntu",
    "sourcesans3", "oswald", "yanonekaffeesatz", "tenorsans", "russoone", "philosopher",
    "comfortaa", "unbounded",
    # антиквы и слэбы
    "ptserif", "notoserif", "lora", "merriweather", "playfairdisplay", "cormorantgaramond",
    "ebgaramond", "alice", "prata", "oldstandardtt", "forum", "kurale", "ledger",
    "sourceserif4", "robotoslab", "kellyslab", "yesevaone",
    # моноширинные
    "ptmono", "robotomono", "firamono", "ibmplexmono", "sourcecodepro", "jetbrainsmono",
    # рукописные
    "caveat", "marckscript", "badscript", "neucha", "amaticsc", "pacifico", "lobster",
    "poiretone", "underdog",
    # декоративные
    "ruslandisplay", "stalinistone", "pressstart2p", "seymourone",
]

PRETRAINED = ["convnext_tiny.fb_in22k_ft_in1k", "lcnet_050.ra2_in1k"]


def download_file(url: str, dst: Path, retries: int = 5, chunk: int = 1 << 20) -> Path:
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Пишем во временный файл, чтобы оборванная загрузка не выглядела готовой, и докачиваем его.
    tmp = dst.with_name(dst.name + ".part")
    for attempt in range(retries):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {**HEADERS, **({"Range": f"bytes={done}-"} if done else {})}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                if done and r.status_code != 206:
                    done = 0
                total = int(r.headers.get("content-length", 0)) + done
                with open(tmp, "ab" if done else "wb") as f, tqdm(
                        total=total, initial=done, unit="B", unit_scale=True, desc=dst.name, leave=False) as bar:
                    for part in r.iter_content(chunk):
                        f.write(part)
                        bar.update(len(part))
            tmp.rename(dst)
            return dst
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return dst


def download_corpora(names=CORPORA, dst: Path = DATA / "corpora") -> list[Path]:
    """Корпуса Leipzig: из архива берём только файл предложений (id<TAB>текст)."""
    out = []
    for name in names:
        txt = dst / f"{name}-sentences.txt"
        if not txt.exists():
            archive = download_file(LEIPZIG_URL.format(name), dst / f"{name}.tar.gz")
            with tarfile.open(archive) as tar:
                member = next(m for m in tar.getmembers() if m.name.endswith("-sentences.txt"))
                member.name = txt.name
                tar.extract(member, dst, filter="data")
            archive.unlink()
        out.append(txt)
    return out


def download_fonts(families=FONT_FAMILIES, dst: Path = DATA / "fonts") -> list[str]:
    """Шрифты и их лицензии из google/fonts. Возвращает семейства, которых нет в репозитории."""
    tree = requests.get(GF_TREE_URL, headers=HEADERS, timeout=120).json()
    if tree.get("truncated"):
        raise RuntimeError("GitHub вернул неполное дерево репозитория google/fonts")

    # Берём только файлы верхнего уровня папки семейства: static/ дублирует вариативные шрифты.
    files: dict[str, list[str]] = {}
    for node in tree["tree"]:
        parts = node["path"].split("/")
        if len(parts) == 3 and parts[0] in ("ofl", "apache", "ufl") and (
                parts[2].lower().endswith((".ttf", ".otf")) or parts[2] in FONT_LICENSES):
            files.setdefault(parts[1], []).append(node["path"])

    for family in tqdm(families, desc="fonts"):
        for path in files.get(family, []):
            download_file(GF_RAW_URL.format(quote(path)), dst / family / Path(path).name)
    return [f for f in families if f not in files]


def download_pretrained(names=PRETRAINED) -> None:
    """Веса ImageNet из timm, кэшируются в ~/.cache/huggingface."""
    import timm

    for name in names:
        timm.create_model(name, pretrained=True)


def summary(root: Path = DATA) -> None:
    """Сколько файлов и мегабайт лежит в каждой подпапке data/."""
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in sub.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files) / 1e6
        print(f"{sub.name:<12} {len(files):>7} файлов {size:>10.1f} МБ")
