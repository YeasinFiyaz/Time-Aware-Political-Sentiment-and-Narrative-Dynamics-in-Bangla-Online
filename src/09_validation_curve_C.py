import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import validation_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from config import OUT_DIR, RANDOM_SEED

def main():
    df = pd.read_csv(OUT_DIR / "political_sentiment_clean.csv")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    param_range = np.logspace(-3, 2, 8)  # 1e-3 to 1e2
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    train_scores, val_scores = validation_curve(
        pipe, X, y,
        param_name="clf__C",
        param_range=param_range,
        cv=cv,
        scoring="f1_macro",     # ✅ robust for multiclass string labels
        n_jobs=-1,
        error_score="raise"     # ✅ stop if fold fails
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    out = pd.DataFrame({
        "C": param_range,
        "train_macro_f1": train_mean,
        "val_macro_f1": val_mean
    })
    out.to_csv(OUT_DIR / "validation_curve_C.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.semilogx(param_range, train_mean, marker="o", label="Train Macro-F1")
    plt.semilogx(param_range, val_mean, marker="o", label="CV Macro-F1")
    plt.title("Validation Curve (C) for Logistic Regression")
    plt.xlabel("C (log scale)")
    plt.ylabel("Macro-F1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "validation_curve_C.png", dpi=300)
    plt.close()

    print("✅ Saved:")
    print(" - validation_curve_C.csv")
    print(" - validation_curve_C.png")

if __name__ == "__main__":
    main()
