import pytesseract
from PIL import Image
import re
import pickle
import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
from PIL import ImageEnhance, ImageFilter
import torch
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification

def extract_info(image_path):
    # Load image and extract text with preprocessing
    image = Image.open(image_path)
    image = image.convert("L")  # Grayscale
    image = ImageEnhance.Contrast(image).enhance(2)  # Boost contrast
    text = pytesseract.image_to_string(image).lower()
    print("Raw Text:", text)  # Debug
    lines = text.split("\n")
    print("First 5 Lines:", lines[0:5])  # Debug

    # Initialize results
    result = {"date": None, "amount": None, "vendor": None, "category": None, "tax_deduction": None}

    # Regex patterns
    date_pattern = r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    amount_pattern = r"(?:\$|USD)?\s?\d+\.?\d{0,2}\s?(?:USD)?"
    vendor_pattern = r"(?:\w+(?:\s\w+)+)(?=\s|$)"

    # Extract date
    date_match = re.search(date_pattern, text)
    if date_match:
        result["date"] = date_match.group(0)

    # Extract amount
    amount_matches = []
    for line in lines:
        total_match = re.search(r"^(total|balance|amount)\s*[:\s]*(" + amount_pattern + ")", line, re.IGNORECASE)
        if total_match:
            amount_matches.append(total_match.group(2))
        sub_total_match = re.search(r"^sub-total\s*[:\s]*(" + amount_pattern + ")", line, re.IGNORECASE)
        if sub_total_match and not amount_matches:
            amount_matches.append(sub_total_match.group(1))

    if amount_matches:
        result["amount"] = max(amount_matches, key=lambda x: float(x.replace("$", "").strip() or "0"), default=None)
        if result["amount"]:
            result["amount"] = "$" + result["amount"].replace("$", "").strip()

    # Extract vendor
    first_five_lines = "\n".join(lines[0:5])
    print("First 5 Lines Joined:", first_five_lines)  # Debug
    vendor_match = re.search(vendor_pattern, first_five_lines)
    if not vendor_match and any(keyword in text for keyword in ["hotel", "restaurant", "bar", "inc", "co"]):
        vendor_match = re.search(vendor_pattern, text)
    if vendor_match:
        result["vendor"] = vendor_match.group(0).capitalize()
    else:
        print("Vendor Match Failed - Pattern:", vendor_pattern, "Text:", first_five_lines)

    # Categorize document
    result["category"] = categorize_document_ml(text)

    # Tax deduction logic
    result["tax_deduction"] = get_tax_deduction(result["amount"], result["category"], result["vendor"])

    return result, text

def categorize_document_ml(text):
    try:
        if not os.path.exists("classifier.pkl") or not os.path.exists("vectorizer.pkl"):
            print("One or more model files missing. Using fallback categorization.")
            return categorize_document(text)
        with open("classifier.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        X = vectorizer.transform([text])
        prediction = classifier.predict(X)[0]
        print(f"ML Prediction: {prediction}")  # Debug
        return prediction
    except FileNotFoundError:
        print("One or more model files missing. Using fallback categorization.")
        return categorize_document(text)
    except ValueError as e:
        print(f"Feature mismatch error: {e}. Using fallback categorization.")
        return categorize_document(text)

def categorize_document(text):
    text = text.lower()
    # More specific rules to distinguish invoices
    if "receipt" in text or "cash" in text or ("total" in text and not "due date" in text):
        return "Receipt"
    elif "invoice" in text or "due date" in text or "bill to" in text or ("total" in text and "due" in text):
        return "Invoice"
    else:
        return "Other"

def get_tax_deduction(amount, category, vendor):
    try:
        amount_value = float(amount.replace("$", "").replace("USD", "").strip()) if amount else 0.0
        if amount_value > 100:
            return "Potential deduction (Business Expense)"
        elif vendor and vendor.lower() in ["amazon", "office depot"]:
            return "Potential deduction (Office Supplies)"
        else:
            return "Not eligible"
    except ValueError:
        return "Unknown"

if __name__ == "__main__":
    image_path = "sample_receipt.jpg"
    info, raw_text = extract_info(image_path)
    print("Extracted Info:", info)
    print("Raw Text:", raw_text)