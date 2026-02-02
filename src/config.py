from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

BANGAL_MEDIA_CSV = DATA_DIR / "BanglaMedia.csv"
POLITICAL_SENTIMENT_XLSX = DATA_DIR / "Political sentiment analysis bengali dataset.xlsx"

RANDOM_SEED = 42