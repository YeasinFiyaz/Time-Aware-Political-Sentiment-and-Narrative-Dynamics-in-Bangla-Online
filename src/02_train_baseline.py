import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from config import OUT_DIR, RANDOM_SEED

def main():
    df = pd.read_csv(OUT_DIR / "political_sentiment_clean.csv")
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n✅ Baseline Evaluation")
    print(classification_report(y_test, preds, digits=4))

    out_model = OUT_DIR / "sentiment_baseline.joblib"
    joblib.dump(model, out_model)
    print("✅ Saved model:", out_model)
    print("\nNext: python src/03_apply_to_banglamedia.py")

if __name__ == "__main__":
    main()