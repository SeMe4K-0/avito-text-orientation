import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Пути можно переопределить, если данные лежат не в репозитории.
DATA = Path(os.environ.get("ORIENT_DATA", ROOT / "data"))
TEST_IMAGES = Path(os.environ.get("ORIENT_TEST", ROOT / "test" / "test" / "images"))
