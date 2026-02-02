import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config import OUT_DIR, RANDOM_SEED

def main():
    df = pd.read_csv(OUT_DIR / "political_sentiment_clean.csv")
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)

    models = {
        "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "MultinomialNB": MultinomialNB(),
        "SGD_Log": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-3)
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro")

        rows.append({"model": name, "accuracy": acc, "macro_f1": macro_f1})

        print("\n==============================")
        print("Model:", name)
        print(classification_report(y_test, pred, digits=4))

    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # Plot comparison (Accuracy + Macro-F1)
    plt.figure(figsize=(9, 4))
    x = np.arange(len(res))
    plt.bar(x - 0.2, res["accuracy"], width=0.4, label="Accuracy")
    plt.bar(x + 0.2, res["macro_f1"], width=0.4, label="Macro-F1")
    plt.xticks(x, res["model"])
    plt.ylim(0, 1)
    plt.title("Model Comparison on Political Sentiment Dataset")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=300)
    plt.close()

    print("\n✅ Saved:")
    print(" - model_comparison.csv")
    print(" - model_comparison.png")

if __name__ == "__main__":
    main()