import pandas as pd
import joblib
from config import OUT_DIR

def main():
    model = joblib.load(OUT_DIR / "sentiment_baseline.joblib")
    df = pd.read_csv(OUT_DIR / "banglamedia_clean.csv")

    df["pred_sentiment"] = model.predict(df["text"].astype(str))

    out_path = OUT_DIR / "banglamedia_with_sentiment.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("✅ Saved:", out_path)
    print("\nSentiment counts:\n", df["pred_sentiment"].value_counts())

    print("\nNext (optional): python src/04_topic_modeling_lda.py")

if __name__ == "__main__":
    main()