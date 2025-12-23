import streamlit as st
from extract_info import extract_info
from PIL import Image
import pandas as pd
import os
import csv
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Financial Document Analyzer")
st.title("Financial Document Analyzer")

@st.cache_resource
def cached_extract(file):
    return extract_info(file)

uploaded_file = st.file_uploader("Upload a receipt or invoice", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Document", width=300)

    try:
        info, raw_text = cached_extract(uploaded_file)
    except Exception as e:
        st.error("Extraction failed")
        st.exception(e)
        st.stop()

    st.subheader("Extracted Information")
    st.write(f"Date: {info.get('date', 'Not found')}")
    st.write(f"Amount: {info.get('amount', 'Not found')}")
    st.write(f"Vendor: {info.get('vendor', 'Not found')}")
    st.write(f"Category: {info.get('category', 'Not found')}")
    st.write(f"Tax Deduction: {info.get('tax_deduction', 'Not found')}")

    file_exists = os.path.isfile("spending_data.csv")
    with open("spending_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Amount", "Vendor", "Category", "Tax_Deduction"])
        writer.writerow([
            info.get("date", "Unknown"),
            info.get("amount", "0.00"),
            info.get("vendor", "Unknown"),
            info.get("category", "Unknown"),
            info.get("tax_deduction", "Unknown")
        ])

if st.button("Show Spending Patterns"):
    if os.path.exists("spending_data.csv"):
        df = pd.read_csv("spending_data.csv")
        df["Amount"] = df["Amount"].replace(r"[^\d.]", "", regex=True).astype(float)
        st.bar_chart(df.groupby("Vendor")["Amount"].sum())
    else:
        st.warning("No spending data available")

if not os.path.exists("classifier.pkl") or not os.path.exists("vectorizer.pkl"):
    st.warning("Model files missing. Retrain?")
    if st.button("Retrain Model"):
        df = pd.read_csv("/Users/shriyajain/Downloads/MLSampleDatasetProj/noisy_dataset_v5_10k.csv")
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]
        model = LogisticRegression(max_iter=1000, C=0.01)
        model.fit(X, y)
        pickle.dump(model, open("classifier.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
        st.success("Model retrained successfully")
