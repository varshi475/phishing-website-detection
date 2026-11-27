
import re
import ipaddress
import pandas as pd
import numpy as np

from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------
# STEP 1: EXTRACT URL FEATURES
# --------------------------------

def extract_features(url):

    url = str(url).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "banking", "signin", "password",
        "confirm", "wallet", "paypal"
    ]

    url_lower = url.lower()

    features = [
        len(url),
        len(hostname),
        url.count("."),
        url.count("-"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        url.count("%"),
        url.count("//"),
        len(path),
        len(query),
        len(hostname.split(".")),
        has_ip,
        1 if parsed.scheme == "https" else 0,
        1 if "@" in url else 0,
        1 if "-" in hostname else 0,
        1 if re.search(r"\d", hostname) else 0,
        sum(word in url_lower for word in suspicious_words)
    ]

    return features

# --------------------------------
# STEP 2: LOAD DATASET
# --------------------------------

print("\nLoading phishing website dataset...")

df = pd.read_csv(r"C:\Users\janan\OneDrive\Desktop\PhishingWebsiteDetection\datasets\Phishing URL.csv")

# Find URL column
url_column = next(
    (c for c in df.columns if c.strip().lower() in ["url", "urls", "website"]),
    None
)

# Find label column
label_column = next(
    (c for c in df.columns if c.strip().lower() in ["label", "type", "class", "status"]),
    None
)

if url_column is None or label_column is None:
    raise ValueError(
        "Dataset must contain URL and Label columns. "
        "Please check your CSV column names."
    )

df = df[[url_column, label_column]].dropna()
df.columns = ["url", "label"]

# --------------------------------
# STEP 3: CONVERT LABELS
# --------------------------------

def convert_label(label):

    value = str(label).strip().lower()

    legitimate_labels = [
        "good", "legitimate", "benign", "safe", "0"
    ]

    phishing_labels = [
        "bad", "phishing", "malicious", "unsafe", "1"
    ]

    if value in legitimate_labels:
        return 0

    elif value in phishing_labels:
        return 1

    else:
        return None

df["label"] = df["label"].apply(convert_label)
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

if df["label"].nunique() != 2:
    raise ValueError(
        "Dataset must contain both legitimate and phishing URLs."
    )

print("Total URLs:", len(df))
print("Legitimate URLs:", (df["label"] == 0).sum())
print("Phishing URLs:", (df["label"] == 1).sum())

# --------------------------------
# STEP 4: FEATURE EXTRACTION
# --------------------------------

print("\nExtracting URL features...")

X = np.array([
    extract_features(url) for url in df["url"]
])

y = df["label"].values

# --------------------------------
# STEP 5: SPLIT DATASET
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# STEP 6: TRAIN MODEL
# --------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------
# STEP 7: MODEL EVALUATION
# --------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------
# STEP 8: PREDICT WEBSITE
# --------------------------------

print("\n====================================")
print(" PHISHING WEBSITE DETECTION SYSTEM")
print("====================================")

while True:

    user_url = input(
        "\nEnter website URL (or type exit): "
    ).strip()

    if user_url.lower() == "exit":
        print("Thank you for using the system!")
        break

    if not user_url:
        print("Please enter a valid URL.")
        continue

    features = np.array([extract_features(user_url)])

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    phishing_probability = probabilities[
        list(model.classes_).index(1)
    ]

    print("\nWebsite URL:", user_url)

    if prediction == 1:
        print("RESULT: PHISHING WEBSITE")
        print("WARNING: This URL has suspicious characteristics.")
    else:
        print("RESULT: LEGITIMATE WEBSITE")
        print("No phishing indicators detected by the model.")

    print(
        "Phishing probability:",
        round(phishing_probability * 100, 2),
        "%"
    )

    print("------------------------------------")