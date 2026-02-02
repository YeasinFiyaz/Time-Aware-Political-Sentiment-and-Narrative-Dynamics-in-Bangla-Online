import re
import pandas as pd
from config import BANGAL_MEDIA_CSV, POLITICAL_SENTIMENT_XLSX, OUT_DIR

def normalize_colname(c: str) -> str:
    return re.sub(r"\s+", "_", str(c).strip().lower())

def guess_text_column(columns):
    candidates = ["text", "comment", "content", "sentence", "review", "post", "status"]
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for c in cols:
        if any(k in c for k in ["text", "comment", "content", "sentence", "review", "post"]):
            return c
    return None

def guess_label_column(columns):
    candidates = ["label", "sentiment", "class", "polarity"]
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for c in cols:
        if any(k in c for k in ["label", "sentiment", "polarity", "class"]):
            return c
    return None

def clean_text(s):
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def map_sentiment(y):
    """
    Convert labels to 3-class: negative / neutral / positive
    Works with numeric (0/1/2 or 1..5) and common strings.
    """
    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.Series(y.dropna().unique()).tolist())
        if len(uniq) >= 5 and min(uniq) >= 1 and max(uniq) <= 5:
            def f(v):
                if v <= 2: return "negative"
                if v == 3: return "neutral"
                return "positive"
            return y.apply(f)
        if set(uniq).issubset({0, 1, 2}):
            return y.map({0: "negative", 1: "neutral", 2: "positive"})
        median = pd.Series(y).median()
        def f(v):
            if v < median: return "negative"
            if v == median: return "neutral"
            return "positive"
        return y.apply(f)

    ys = y.astype(str).str.lower().str.strip()
    def f(v):
        if any(k in v for k in ["neg", "bad", "anger", "sad"]): return "negative"
        if any(k in v for k in ["neu", "mid", "ok"]): return "neutral"
        if any(k in v for k in ["pos", "good", "happy", "support"]): return "positive"
        return "neutral"
    return ys.apply(f)

def main():
    df_media = pd.read_csv(BANGAL_MEDIA_CSV)
    df_pol = pd.read_excel(POLITICAL_SENTIMENT_XLSX)

    df_media.columns = [normalize_colname(c) for c in df_media.columns]
    df_pol.columns = [normalize_colname(c) for c in df_pol.columns]

    media_text_col = guess_text_column(df_media.columns)
    pol_text_col = guess_text_column(df_pol.columns)
    pol_label_col = guess_label_column(df_pol.columns)

    if media_text_col is None:
        raise ValueError(f"❌ Could not detect text column in BanglaMedia. Columns: {list(df_media.columns)}")
    if pol_text_col is None or pol_label_col is None:
        raise ValueError(
            "❌ Could not detect text/label columns in Political sentiment dataset.\n"
            f"Columns: {list(df_pol.columns)}\n"
            "✅ Rename columns to something like: text, sentiment"
        )

    media_clean = pd.DataFrame({"text": df_media[media_text_col].astype(str).map(clean_text)})
    media_clean = media_clean[media_clean["text"].str.len() > 1]

    pol_clean = pd.DataFrame({
        "text": df_pol[pol_text_col].astype(str).map(clean_text),
        "label_raw": df_pol[pol_label_col]
    }).dropna()
    pol_clean["label"] = map_sentiment(pol_clean["label_raw"])
    pol_clean = pol_clean[pol_clean["text"].str.len() > 1]

    media_out = OUT_DIR / "banglamedia_clean.csv"
    pol_out = OUT_DIR / "political_sentiment_clean.csv"

    media_clean.to_csv(media_out, index=False, encoding="utf-8-sig")
    pol_clean.to_csv(pol_out, index=False, encoding="utf-8-sig")

    print("✅ Saved:", media_out, "shape=", media_clean.shape)
    print("✅ Saved:", pol_out, "shape=", pol_clean.shape)
    print("\nNext: python src/02_train_baseline.py")

if __name__ == "__main__":
    main()