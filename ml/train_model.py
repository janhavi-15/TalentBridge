import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ========================================
# LOAD DATASET
# ========================================

df = pd.read_csv("training_data.csv")

print("Dataset loaded")
print("Original records:", len(df))


# ========================================
# CLEAN DATA
# ========================================

# Remove rows where ML-required columns are missing
df = df.dropna(
    subset=["title", "description", "skills", "field"]
)

# Convert to text
for column in ["title", "description", "skills", "field"]:
    df[column] = df[column].astype(str).str.strip()

# Remove completely empty rows
df = df[
    (df["title"] != "") &
    (df["description"] != "") &
    (df["skills"] != "") &
    (df["field"] != "")
]


print("Valid records:", len(df))


# ========================================
# CREATE TRAINING TEXT
# ========================================

X = (
    "TITLE " + df["title"] +
    " DESCRIPTION " + df["description"] +
    " SKILLS " + df["skills"]
)

y = df["field"]


# ========================================
# CHECK FIELD DISTRIBUTION
# ========================================

print("\nField distribution:")
print(y.value_counts())


# ========================================
# TRAIN / TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ========================================
# MACHINE LEARNING PIPELINE
# ========================================

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )

])


# ========================================
# TRAIN MODEL
# ========================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# ========================================
# TEST MODEL
# ========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("\n================================")
print("MODEL ACCURACY")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ========================================
# SAVE MODEL
# ========================================

joblib.dump(
    model,
    "job_field_model.pkl"
)

print("\n================================")
print("MODEL SAVED")
print("================================")

print("ml/job_field_model.pkl")