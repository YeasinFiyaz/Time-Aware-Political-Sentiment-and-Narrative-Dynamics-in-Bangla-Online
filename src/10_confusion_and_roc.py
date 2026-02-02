import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import OUT_DIR, RANDOM_SEED

def main():
    df = pd.read_csv(OUT_DIR / "political_sentiment_clean.csv")
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    classes = ["negative", "neutral", "positive"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    cm_df.to_csv(OUT_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(6,5))
    plt.imshow(cm, aspect="auto")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.colorbar(label="Count")
    plt.title("Confusion Matrix (TF-IDF + Logistic Regression)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # -----------------------------
    # ROC Curves (One-vs-Rest)
    # -----------------------------
    probs = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(8,6))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{c} (AUC={roc_auc:.3f})")

    plt.plot([0,1], [0,1], linestyle="--", alpha=0.6)
    plt.title("ROC Curves (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=300)
    plt.close()

    print("✅ Saved:")
    print(" - confusion_matrix.csv / confusion_matrix.png")
    print(" - roc_curves.png")

if __name__ == "__main__":
    main()
