"""Скачивание внешних данных и предобученных весов.

Все источники открытые и не требуют токенов. Функции можно перезапускать:
уже скачанное пропускается, оборванные загрузки докачиваются.
"""
import re
import tarfile
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from tqdm.auto import tqdm

from .paths import DATA

# Wikimedia требует осмысленный User-Agent со ссылкой на проект.
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

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Корневые категории Commons с фото русского текста в реальных сценах.
COMMONS_SEEDS = [
    "Russian-language signs", "Signs in Russia", "Shop signs in Russia", "Advertisements in Russia",
    "Billboards in Russia", "Signs in Moscow", "Shop signs in Moscow", "Signs in Saint Petersburg",
    "Price tags",
]
OPEN_LICENSE = re.compile(r"^(CC0|CC[ -]BY|Public domain|PD)", re.IGNORECASE)

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


def _commons(params: dict) -> dict:
    for attempt in range(5):
        try:
            r = requests.get(COMMONS_API, params={**params, "format": "json"}, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Commons API не отвечает: {params}")


def commons_manifest(path: Path = DATA / "commons_manifest.csv", seeds=COMMONS_SEEDS, depth: int = 3,
                     max_categories: int = 600, width: int = 1024) -> pd.DataFrame:
    """Список фото из категорий Commons (обход в ширину) с адресом превью, автором и лицензией.

    Список сохраняется в репозиторий: состав категорий со временем меняется,
    а скачивание по сохранённому списку даёт тот же набор фото.
    """
    if path.exists():
        return pd.read_csv(path)
    titles, seen, queue = set(), set(), [(f"Category:{s}", 0) for s in seeds]
    with tqdm(desc="categories") as bar:
        while queue and len(seen) < max_categories:
            cat, d = queue.pop(0)
            if cat in seen:
                continue
            seen.add(cat)
            bar.update(1)
            cont = {}
            while True:
                r = _commons({"action": "query", "list": "categorymembers", "cmtitle": cat,
                              "cmtype": "file|subcat", "cmlimit": 500, **cont})
                for m in r["query"]["categorymembers"]:
                    if m["ns"] == 6 and m["title"].lower().endswith((".jpg", ".jpeg", ".png")):
                        titles.add(m["title"])
                    elif m["ns"] == 14 and d < depth:
                        queue.append((m["title"], d + 1))
                if "continue" not in r:
                    break
                cont = r["continue"]

    rows, titles = [], sorted(titles)
    for i in tqdm(range(0, len(titles), 50), desc="imageinfo"):
        r = _commons({"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo",
                      "iiprop": "url|extmetadata", "iiurlwidth": width,
                      "iiextmetadatafilter": "LicenseShortName|Artist"})
        for page in r["query"]["pages"].values():
            info = page.get("imageinfo", [{}])[0]
            meta = info.get("extmetadata", {})
            rows.append({
                "pageid": page.get("pageid"), "title": page["title"],
                "url": info.get("thumburl") or info.get("url"),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "artist": re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip(),
                "source": info.get("descriptionurl"),
            })
    df = pd.DataFrame(rows).dropna(subset=["pageid", "url"])
    df = df[df.license.str.match(OPEN_LICENSE)].sort_values("pageid").reset_index(drop=True)
    df["pageid"] = df.pageid.astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def download_commons(manifest: pd.DataFrame, dst: Path = DATA / "photos" / "commons") -> list[Path]:
    """Качает превью фото по списку. Недоступные файлы пропускает и сообщает их число."""
    out, failed = [], 0
    for row in tqdm(manifest.itertuples(), total=len(manifest), desc="commons"):
        try:
            out.append(download_file(row.url, dst / f"{row.pageid}.jpg"))
        except requests.HTTPError:
            failed += 1
    if failed:
        print(f"не скачалось: {failed}")
    return out


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
