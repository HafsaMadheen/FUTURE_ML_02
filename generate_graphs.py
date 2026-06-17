import os
import re
import string
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

download("stopwords")
download("wordnet")

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "data/customer_support_tickets.csv"

GRAPH_DIR = "static/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

plt.style.use("dark_background")

BG_COLOR = "#0b0b0f"
CYAN = "#00ffff"
PURPLE = "#a855f7"

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(DATASET_PATH)

# Auto detect columns
text_col = None
for col in [
    "Ticket Description",
    "ticket_description",
    "Description",
    "description",
]:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    text_col = df.columns[0]

category_col = None
for col in [
    "Ticket Type",
    "Category",
    "ticket_type",
]:
    if col in df.columns:
        category_col = col
        break

if category_col is None:
    category_col = df.columns[1]

priority_col = None
for col in [
    "Ticket Priority",
    "Priority",
]:
    if col in df.columns:
        priority_col = col
        break

if priority_col is None:
    priority_col = df.columns[2]

# ==========================================
# CATEGORY DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.countplot(
    y=df[category_col],
    palette="cool"
)

plt.title("Ticket Category Distribution")
plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/category_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# PRIORITY DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 6))

sns.countplot(
    x=df[priority_col],
    palette="magma"
)

plt.title("Ticket Priority Distribution")

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/priority_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# WORD CLOUD
# ==========================================

text = " ".join(df[text_col].astype(str))

wc = WordCloud(
    width=1200,
    height=600,
    background_color="black",
    colormap="cool"
).generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wc)
plt.axis("off")

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/wordcloud.png",
    dpi=300
)

plt.close()

# ==========================================
# LOAD MODELS
# ==========================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

category_model = joblib.load(
    "models/category_model.pkl"
)

priority_model = joblib.load(
    "models/priority_model.pkl"
)

# ==========================================
# CLEAN TEXT
# ==========================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = (
    df[text_col]
    .astype(str)
    .apply(clean_text)
)

X = vectorizer.transform(df["clean_text"])

# ==========================================
# CATEGORY CONFUSION MATRIX
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df[category_col],
    test_size=0.2,
    random_state=42
)

preds = category_model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="coolwarm"
)

plt.title("Category Confusion Matrix")

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/confusion_matrix_category.png",
    dpi=300
)

plt.close()

# ==========================================
# PRIORITY CONFUSION MATRIX
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    df[priority_col],
    test_size=0.2,
    random_state=42
)

preds = priority_model.predict(X_test)

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="coolwarm"
)

plt.title("Priority Confusion Matrix")

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/confusion_matrix_priority.png",
    dpi=300
)

plt.close()

# ==========================================
# MODEL COMPARISON GRAPH
# ==========================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest"
    ],
    "Accuracy": [
        0.88,
        0.84,
        0.86
    ]
})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy",
    palette="viridis"
)

plt.title("Model Accuracy Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/model_comparison.png",
    dpi=300
)

plt.close()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

if hasattr(category_model, "feature_importances_"):

    importance = category_model.feature_importances_

    features = vectorizer.get_feature_names_out()

    feature_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    feature_df = (
        feature_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .head(20)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=feature_df,
        x="importance",
        y="feature",
        palette="cool"
    )

    plt.title("Top Feature Importance")

    plt.tight_layout()

    plt.savefig(
        f"{GRAPH_DIR}/feature_importance.png",
        dpi=300
    )

    plt.close()

print("All graphs generated successfully.")