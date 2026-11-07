"""Train and evaluate the spam classifier, then save the best pipeline."""
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import clean_text

DATA_PATH = "data/enron_spam_data.zip"
MODEL_PATH = "models/spam_model.joblib"
METRICS_PATH = "models/metrics.json"

# ---- Part 1: load and explore ----
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df["Spam/Ham"].value_counts(), "\n")
for label in ("spam", "ham"):
    sample = df[df["Spam/Ham"] == label].iloc[0]
    print(f"--- Example {label.upper()} ---\nSubject: {sample['Subject']}\n{str(sample['Message'])[:200]}\n")

# ---- Part 2: preprocessing ----
df["Subject"] = df["Subject"].fillna("")
df["Message"] = df["Message"].fillna("")
df["text"] = (df["Subject"] + " " + df["Message"]).map(clean_text)
df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")
df["label"] = (df["Spam/Ham"] == "spam").astype(int)
print("After cleaning:", df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# ---- Part 3: models ----
def make(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000,
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])

candidates = {
    "Naive Bayes": make(MultinomialNB(alpha=0.1)),
    "Logistic Regression": make(LogisticRegression(max_iter=1000, C=10)),
}

results, best_name, best_f1, best_model = {}, None, -1, None
for name, model in candidates.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    m = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    results[name] = m
    print(f"=== {name} ===")
    print(classification_report(y_test, pred, target_names=["Not Spam", "Spam"]))
    print("Confusion matrix [[TN FP],[FN TP]]:", m["confusion_matrix"], "\n")
    if m["f1"] > best_f1:
        best_name, best_f1, best_model = name, m["f1"], model

print("Best model:", best_name)
joblib.dump(best_model, MODEL_PATH, compress=3)
results["best_model"] = best_name
results["test_size"] = int(len(y_test))
with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2)

# ---- Unseen example check ----
tests = [
    "Congratulations! You won a $1000 gift card. Click http://win-now.com to claim your free prize today!!!",
    "Hi team, attaching the meeting notes from yesterday. Let's review the Q3 schedule on Monday at 10am.",
]
for t in tests:
    p = best_model.predict_proba([clean_text(t)])[0][1]
    print(f"{'SPAM' if p > 0.5 else 'NOT SPAM'} ({p:.2%}): {t[:70]}")
