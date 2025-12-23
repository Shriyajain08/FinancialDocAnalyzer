import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from extract_info import extract_info
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os

class FinancialDocAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Document Analyzer")
        self.root.geometry("600x400")

        # Image display
        self.image_label = tk.Label(self.root, text="No image loaded")
        self.image_label.pack(pady=10)

        # Buttons
        self.upload_btn = tk.Button(self.root, text="Upload Document", command=self.upload_image)
        self.upload_btn.pack(pady=5)
        self.plot_btn = tk.Button(self.root, text="Show Spending Patterns", command=self.plot_spending)
        self.plot_btn.pack(pady=5)

        # Results display
        self.result_text = tk.Text(self.root, height=6, width=50)
        self.result_text.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            try:
                # Display image
                img = Image.open(file_path)
                img = img.resize((200, 200), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk, text="")
                self.image_label.image = img_tk

                # Extract info
                info, _ = extract_info(file_path)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Date: {info['date'] or 'Not found'}\n")
                self.result_text.insert(tk.END, f"Amount: {info['amount'] or 'Not found'}\n")
                self.result_text.insert(tk.END, f"Vendor: {info['vendor'] or 'Not found'}\n")
                self.result_text.insert(tk.END, f"Category: {info['category'] or 'Not found'}\n")
                self.result_text.insert(tk.END, f"Tax Deduction: {info['tax_deduction'] or 'Not found'}\n")

                # Save to CSV
                self.save_to_csv(info)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")

    def save_to_csv(self, info):
        file_exists = os.path.isfile("spending_data.csv")
        with open("spending_data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Date", "Amount", "Vendor", "Category", "Tax_Deduction"])
            writer.writerow([
                info["date"] or "Unknown",
                info["amount"] or "0.00",
                info["vendor"] or "Unknown",
                info["category"] or "Unknown",
                info["tax_deduction"] or "Unknown"
            ])

    def plot_spending(self):
        try:
            # Read CSV with specific columns
            df = pd.read_csv("spending_data.csv", usecols=["Date", "Amount", "Vendor", "Category", "Tax_Deduction"], on_bad_lines='skip')
            # Clean amount for plotting
            df["Amount"] = df["Amount"].replace(r"[^\d.]", "", regex=True).astype(float)
            # Plot by vendor
            plt.figure(figsize=(8, 6))
            df.groupby("Vendor")["Amount"].sum().plot(kind="bar")
            plt.title("Spending by Vendor")
            plt.ylabel("Total Amount ($)")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialDocAnalyzer(root)
    root.mainloop()