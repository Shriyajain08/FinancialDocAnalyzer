import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the dataset
df = pd.read_csv("/Users/shriyajain/Downloads/ProjCopy/FinancialDocumentDatasetCopy.csv")

# Ensure balanced classes (if dataset allows)
print("Class distribution before balancing:", df['label'].value_counts())
# If imbalance is significant, you might want to undersample or oversample (e.g., using imbalanced-learn)
# For now, we'll proceed with the given data

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

print("Class distribution:", y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify for balance

model = LogisticRegression(max_iter=1000, C=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["Receipt", "Invoice", "Other"])
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Receipt", "Invoice", "Other"], yticklabels=["Receipt", "Invoice", "Other"])
plt.title("Confusion Matrix Heatmap")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png")
print("Confusion matrix heatmap saved as 'confusion_matrix_heatmap.png'")
plt.show()

# Save model and vectorizer in the format expected by oldextractinfo.py
with open("classifier.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("Model saved as 'classifier.pkl' and vectorizer as 'vectorizer.pkl'")