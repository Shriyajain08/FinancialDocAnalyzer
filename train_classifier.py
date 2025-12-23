# train_classifier.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.base import TransformerMixin
from scipy.sparse import hstack  # Added missing import

# Custom transformer for text stats
class TextStats(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.array([[len(x.split()), int("total" in x.lower())] for x in X]).astype(float)

# Load dataset
data_path = "/Users/shriyajain/Downloads/MLSampleDatasetProj/FinancialDocumentDataset.csv"
df = pd.read_csv(data_path)

# Prepare data
texts = df["text"].values
labels = df["label"].values

# Split into training, validation, and testing sets (70-10-20)
X_train_texts, X_val_test_texts, y_train, y_val_test = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
X_val_texts, X_test_texts, y_val, y_test = train_test_split(X_val_test_texts, y_val_test, test_size=0.67, random_state=42, stratify=y_val_test)

# Vectorize text with enhanced features
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train = vectorizer.fit_transform(X_train_texts)
X_test = vectorizer.transform(X_test_texts)
stats_train = TextStats().transform(X_train_texts)
stats_test = TextStats().transform(X_test_texts)
X_train_enhanced = hstack([X_train, stats_train])  # Now works with the import
X_test_enhanced = hstack([X_test, stats_test])

# Train classifier
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_train_enhanced, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_enhanced)

# Print metrics
print("=== Evaluation Metrics ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(classifier, X_train_enhanced, y_train, cv=cv)
print(f"\n10-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=df["label"].unique(), yticklabels=df["label"].unique())
plt.savefig("confusion_matrix.png")
plt.close()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(classifier, X_train_enhanced, y_train, cv=5, n_jobs=-1)
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
plt.plot(train_sizes, val_scores.mean(axis=1), label="Cross-Validation Score")
plt.legend()
plt.savefig("learning_curve.png")
plt.close()

# Save model and vectorizer
with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model trained and saved successfully!")