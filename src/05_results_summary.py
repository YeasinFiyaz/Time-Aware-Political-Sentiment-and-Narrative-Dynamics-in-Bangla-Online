import pandas as pd
import matplotlib.pyplot as plt
from config import OUT_DIR

def save_barplot(series, title, filename):
    plt.figure(figsize=(7,4))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(series.name if series.name else "")
    plt.ylabel("Count")
    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=300)
    plt.close()
    print("✅ Saved plot:", out)

def main():
    # Sentiment predictions
    df = pd.read_csv(OUT_DIR / "banglamedia_with_sentiment.csv")

    # Overall sentiment distribution
    sentiment_counts = df["pred_sentiment"].value_counts()
    sentiment_counts.to_csv(OUT_DIR / "sentiment_counts.csv", header=["count"])
    print("\n✅ Sentiment counts:\n", sentiment_counts)

    save_barplot(sentiment_counts, "BanglaMedia Predicted Sentiment Distribution", "sentiment_distribution.png")

    # Topic distribution (LDA)
    df_topic = pd.read_csv(OUT_DIR / "banglamedia_with_topics_lda.csv")
    topic_counts = df_topic["topic_lda"].value_counts().sort_index()
    topic_counts.to_csv(OUT_DIR / "topic_counts.csv", header=["count"])
    print("\n✅ Topic counts (LDA):\n", topic_counts)

    plt.figure(figsize=(9,4))
    topic_counts.plot(kind="bar")
    plt.title("LDA Topic Distribution (BanglaMedia)")
    plt.xlabel("Topic ID")
    plt.ylabel("Count")
    plt.tight_layout()
    out = OUT_DIR / "topic_distribution.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("✅ Saved plot:", out)

    # Topic-wise sentiment distribution
    merged = df_topic.merge(df[["text", "pred_sentiment"]], on="text", how="left")
    ctab = pd.crosstab(merged["topic_lda"], merged["pred_sentiment"], normalize="index")
    ctab.to_csv(OUT_DIR / "topic_sentiment_share.csv")
    print("\n✅ Saved topic_sentiment_share.csv")

    # Plot heatmap-like (simple) using matplotlib
    plt.figure(figsize=(8,5))
    plt.imshow(ctab.values, aspect="auto")
    plt.xticks(range(len(ctab.columns)), ctab.columns)
    plt.yticks(range(len(ctab.index)), ctab.index)
    plt.colorbar(label="Share")
    plt.title("Topic vs Sentiment (Row-normalized share)")
    plt.tight_layout()
    out = OUT_DIR / "topic_vs_sentiment.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("✅ Saved plot:", out)

    print("\n✅ Done. Check outputs/ folder for CSV + PNG files.")

if __name__ == "__main__":
    main()