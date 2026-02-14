import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset (replace with real CSV dataset)
data = {
    "text": [
        "Breaking: Aliens landed in New York!",
        "Celebrity announces shocking discovery.",
        "Government passes new education policy.",
        "NASA launches new Mars mission.",
        "Local man discovers internet still working despite unpaid WiFi bill.","narendra modi is prime minister"
    ],
    "label": ["Fake News", "Fake News", "Real News", "Real News", "Satire","Real News"]
}

df = pd.DataFrame(data)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    return text

df["clean_text"] = df["text"].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("✅ Training Complete")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model + vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
