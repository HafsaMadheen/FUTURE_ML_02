from flask import Flask, render_template, request
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download

download("stopwords")
download("wordnet")

app = Flask(__name__)

# ==================================
# LOAD MODELS
# ==================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

category_model = joblib.load(
    "models/category_model.pkl"
)

priority_model = joblib.load(
    "models/priority_model.pkl"
)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ==================================
# CLEAN TEXT
# ==================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ==================================
# HOME
# ==================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==================================
# PREDICT
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        ticket = request.form["ticket"]

        cleaned = clean_text(ticket)

        vector = vectorizer.transform(
            [cleaned]
        )

        category = (
            category_model.predict(vector)[0]
        )

        priority = (
            priority_model.predict(vector)[0]
        )

        confidence = max(
            category_model.predict_proba(
                vector
            )[0]
        )

        confidence = round(
            confidence * 100,
            2
        )

        return render_template(
            "index.html",
            category=category,
            priority=priority,
            confidence=confidence,
            ticket=ticket
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )

# ==================================
# RUN
# ==================================

if __name__ == "__main__":
    app.run(
        debug=True
    )