import sys
import pandas as pd
from config import BANGAL_MEDIA_CSV, POLITICAL_SENTIMENT_XLSX

def main():
    print("✅ Python:", sys.version)
    print("✅ Pandas:", pd.__version__)
    print("✅ CSV path:", BANGAL_MEDIA_CSV)
    print("✅ XLSX path:", POLITICAL_SENTIMENT_XLSX)

    assert BANGAL_MEDIA_CSV.exists(), f"Missing file: {BANGAL_MEDIA_CSV}"
    assert POLITICAL_SENTIMENT_XLSX.exists(), f"Missing file: {POLITICAL_SENTIMENT_XLSX}"

    df_csv = pd.read_csv(BANGAL_MEDIA_CSV)
    print("✅ BanglaMedia loaded:", df_csv.shape)

    df_xlsx = pd.read_excel(POLITICAL_SENTIMENT_XLSX)
    print("✅ Political sentiment XLSX loaded:", df_xlsx.shape)

    print("\n✅ Setup OK. Next: python src/01_prepare_datasets.py")

if __name__ == "__main__":
    main()