import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from config import OUT_DIR

def main():
    df = pd.read_csv(OUT_DIR / "banglamedia_clean.csv")
    texts = df["text"].astype(str).tolist()

    vectorizer = CountVectorizer(max_df=0.95, min_df=5)
    X = vectorizer.fit_transform(texts)

    n_topics = 10
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    topic_dist = lda.fit_transform(X)
    df["topic_lda"] = topic_dist.argmax(axis=1)

    out_path = OUT_DIR / "banglamedia_with_topics_lda.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("✅ Saved:", out_path)

    words = vectorizer.get_feature_names_out()
    topn = 12
    rows = []
    for k, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-topn:][::-1]
        rows.append({"topic": k, "top_words": ", ".join(words[i] for i in top_idx)})

    pd.DataFrame(rows).to_csv(OUT_DIR / "lda_topic_words.csv", index=False, encoding="utf-8-sig")
    print("✅ Saved:", OUT_DIR / "lda_topic_words.csv")

if __name__ == "__main__":
    main()