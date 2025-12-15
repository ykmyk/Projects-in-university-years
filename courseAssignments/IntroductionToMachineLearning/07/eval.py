#!/usr/bin/env python3
import lzma
import pickle

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from isnt_it_ironic import Dataset
from isnt_it_ironic import main as train_main  # or just re-import the pipeline pieces

# Load the full dataset
full = Dataset()

# Split into train/dev
X_train, X_dev, y_train, y_dev = train_test_split(
    full.data, full.target, test_size=0.2, random_state=42, stratify=full.target
)

# --- recreate your model here exactly as in isnt_it_ironic.py ---
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("count", CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.99,
    )),

    ("tfidf", TfidfTransformer(
        norm="l1",
        sublinear_tf=False,
        smooth_idf=False,
        )),

    ("classifier", LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        fit_intercept=False,
        penalty="l2",
        solver="newton-cg",
        multi_class="multinomial",
    )
)
])

model.fit(X_train, y_train)
pred_dev = model.predict(X_dev)
print("F1 on dev split:", f1_score(y_dev, pred_dev))
