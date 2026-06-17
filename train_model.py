import os
import re
import string
import joblib
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

download('stopwords')
download('wordnet')

os.makedirs("models", exist_ok=True)

DATASET_PATH = "data/customer_support_tickets.csv"

df = pd.read_csv(DATASET_PATH)

print("\nDATASET INFO")
print(df.info())

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ----------- Detect Text Column Automatically ----------
possible_text_cols = [
    "Ticket Description",
    "ticket_description",
    "Description",
    "description",
    "Ticket"
]

text_col = None

for col in possible_text_cols:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    text_col = df.columns[0]

# -------- Detect Category ----------
possible_category_cols = [
    "Ticket Type",
    "Category",
    "ticket_type"
]

category_col = None

for col in possible_category_cols:
    if col in df.columns:
        category_col = col
        break

if category_col is None:
    category_col = df.columns[1]

# -------- Detect Priority ----------
possible_priority_cols = [
    "Ticket Priority",
    "Priority",
    "priority"
]

priority_col = None

for col in possible_priority_cols:
    if col in df.columns:
        priority_col = col
        break

if priority_col is None:
    priority_col = df.columns[2]

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'\d+', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = df[text_col].astype(str).apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

# ==================================================
# CATEGORY CLASSIFICATION
# ==================================================

y_category = df[category_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42
)

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Naive Bayes":
        MultinomialNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

best_model = None
best_accuracy = 0

print("\nCATEGORY CLASSIFICATION\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\n{name}")
    print("Accuracy:", acc)

    print(
        classification_report(
            y_test,
            preds
        )
    )

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

joblib.dump(
    best_model,
    "models/category_model.pkl"
)

print("\nBest Category Model Saved")

# ==================================================
# PRIORITY PREDICTION
# ==================================================

y_priority = df[priority_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42
)

best_model = None
best_accuracy = 0

print("\nPRIORITY PREDICTION\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"\n{name}")
    print("Accuracy:", acc)

    print(
        classification_report(
            y_test,
            preds
        )
    )

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

joblib.dump(
    best_model,
    "models/priority_model.pkl"
)

print("\nBest Priority Model Saved")
