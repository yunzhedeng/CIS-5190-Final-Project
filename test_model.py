import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. Load and clean the dataset
# ============================================================

# Use the Python CSV engine to tolerate malformed lines
df = pd.read_csv(
    "final_headlines.csv",
    engine="python",
    on_bad_lines="skip"
)

# Remove rows with missing headline or label
df = df.dropna(subset=["headline", "label"])

# Basic text cleaning
df["headline"] = df["headline"].astype(str).str.strip()
df = df[df["headline"].str.len() >= 5]

X = df["headline"]
y = df["label"].astype(str)

# ============================================================
# 2. Split out the TEST set ONCE (never touch it during tuning)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,          # 10% held-out test set
    random_state=42,
    stratify=y              # preserve class distribution
)

# ============================================================
# 3. Build the modeling pipeline
# ============================================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True)),
    ("clf", LinearSVC())
])

# ============================================================
# 4. Define hyperparameter search space
#    (kept moderate to avoid overfitting CV)
# ============================================================

param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2), (1,3)],
    "tfidf__min_df": [2, 3, 5],
    "tfidf__max_df": [0.9, 0.95],
    "clf__C": [0.5, 1.0, 2.0, 4.0],
    "clf__class_weight": [None, "balanced"]
}

# ============================================================
# 5. K-fold cross-validation on TRAIN only
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",     # robust metric for class imbalance
    cv=cv,
    n_jobs=-1,              # use all CPU cores
    verbose=2
)

# Run CV-based hyperparameter selection
grid_search.fit(X_train, y_train)

print("\nBest CV macro-F1:", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)

# ============================================================
# 6. Final evaluation on the untouched TEST set
# ============================================================

best_model = grid_search.best_estimator_

test_predictions = best_model.predict(X_test)

print("\nTEST RESULTS")
print(classification_report(y_test, test_predictions, digits=4))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, test_predictions))
