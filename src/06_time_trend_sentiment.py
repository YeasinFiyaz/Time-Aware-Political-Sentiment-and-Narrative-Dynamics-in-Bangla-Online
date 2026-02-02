import pandas as pd
import matplotlib.pyplot as plt
from config import OUT_DIR, DATA_DIR

def main():
    # Load original BanglaMedia to get Date
    raw = pd.read_csv(DATA_DIR / "BanglaMedia.csv")

    # Normalize columns
    raw.columns = raw.columns.str.strip().str.lower()

    # Load sentiment predictions
    pred = pd.read_csv(OUT_DIR / "banglamedia_with_sentiment.csv")

    # Merge on text/comments
    merged = raw.merge(
        pred,
        left_on="comments",
        right_on="text",
        how="inner"
    )

    # Parse date
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])

    # Create time bins
    merged["year"] = merged["date"].dt.year
    merged["year_month"] = merged["date"].dt.to_period("M").astype(str)

    # ===============================
    # 1. Sentiment proportion by year
    # ===============================
    yearly = (
        pd.crosstab(merged["year"], merged["pred_sentiment"], normalize="index")
        .sort_index()
    )
    yearly.to_csv(OUT_DIR / "yearly_sentiment_share.csv")

    plt.figure(figsize=(10,5))
    yearly.plot(kind="line", marker="o")
    plt.title("Yearly Political Sentiment Share in BanglaMedia")
    plt.ylabel("Proportion")
    plt.xlabel("Year")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "yearly_sentiment_trend.png", dpi=300)
    plt.close()

    # ===============================
    # 2. Monthly net sentiment index
    # ===============================
    sentiment_score = {"negative": -1, "neutral": 0, "positive": 1}
    merged["sentiment_score"] = merged["pred_sentiment"].map(sentiment_score)

    monthly = (
        merged.groupby("year_month")["sentiment_score"]
        .mean()
        .reset_index()
    )

    monthly.to_csv(OUT_DIR / "monthly_net_sentiment.csv", index=False)

    plt.figure(figsize=(12,5))
    plt.plot(monthly["year_month"], monthly["sentiment_score"], color="black")
    plt.axhline(0, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.title("Monthly Net Political Sentiment Index (BanglaMedia)")
    plt.ylabel("Net Sentiment (-1 = negative, +1 = positive)")
    plt.xlabel("Year-Month")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "monthly_net_sentiment.png", dpi=300)
    plt.close()

    print("✅ Saved:")
    print(" - yearly_sentiment_share.csv")
    print(" - yearly_sentiment_trend.png")
    print(" - monthly_net_sentiment.csv")
    print(" - monthly_net_sentiment.png")

if __name__ == "__main__":
    main()