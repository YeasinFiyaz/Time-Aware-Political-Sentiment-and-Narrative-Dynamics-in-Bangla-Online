import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import OUT_DIR, RANDOM_SEED

def main():
    df = pd.read_csv(OUT_DIR / "political_sentiment_clean.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    # Stratified CV to keep class distribution balanced in each fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    train_sizes = np.linspace(0.1, 1.0, 8)

    sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="f1_macro",   # ✅ robust for multiclass string labels
        n_jobs=-1,
        error_score="raise"   # ✅ stop if any fold fails (no silent NaNs)
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    out = pd.DataFrame({
        "train_size": sizes,
        "train_macro_f1": train_mean,
        "val_macro_f1": val_mean
    })
    out.to_csv(OUT_DIR / "learning_curve.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(sizes, train_mean, marker="o", label="Train Macro-F1")
    plt.plot(sizes, val_mean, marker="o", label="CV Macro-F1")
    plt.title("Learning Curve (TF-IDF + Logistic Regression)")
    plt.xlabel("Training Samples")
    plt.ylabel("Macro-F1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "learning_curve.png", dpi=300)
    plt.close()

    print("✅ Saved:")
    print(" - learning_curve.csv")
    print(" - learning_curve.png")

if __name__ == "__main__":
    main()
