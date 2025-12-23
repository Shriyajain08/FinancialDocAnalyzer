from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os
import pandas as pd
import numpy as np

# Configure Tesseract path
tesseract_paths = [
    '/opt/homebrew/bin/tesseract',  # Confirmed path
    '/usr/local/bin/tesseract',
    '/usr/bin/tesseract'
]
for path in tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        print(f"Using Tesseract at: {path}")
        break
else:
    print("Tesseract not found. Please install it or set the correct path manually.")
    exit(1)

# Set the input directory
input_dir = "/Users/shriyajain/Downloads/MLSampleDatasetProj/test/new"
output_file = "rvlcdip_text_labels.csv"

data = []

# Recursively walk through subdirectories
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            img_path = os.path.join(root, file)
            print(f"Processing: {img_path}")
            try:
                # Open and preprocess the image
                img = Image.open(img_path).convert("L")  # Grayscale
                img = img.filter(ImageFilter.MedianFilter(size=3))  # Noise reduction
                # Apply enhancements using ImageEnhance
                enhancer_contrast = ImageEnhance.Contrast(img)
                img = enhancer_contrast.enhance(2.5)  # Increase contrast
                enhancer_sharpness = ImageEnhance.Sharpness(img)
                img = enhancer_sharpness.enhance(2)  # Increase sharpness
                # Binary thresholding for better contrast
                img = img.point(lambda x: 0 if x < 128 else 255, '1')
                # Try multiple PSM modes
                for psm in [6, 4, 3, 1]:  # 6: single block, 4: column, 3: auto, 1: automatic page segmentation
                    text = pytesseract.image_to_string(img, config=f'--psm {psm} --oem 3').lower().strip()  # OEM 3: default
                    if text and len(text.split()) > 0:  # Relaxed to at least 1 word
                        break
                label = os.path.basename(root)  # e.g., "form" from test/images/form
                if text and len(text.split()) > 0:
                    data.append({"text": text, "label": label})
                    print(f"Extracted from {img_path}: '{text[:50]}...' (label: {label})")
                else:
                    print(f"No usable text extracted from {img_path} with PSM {psm}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
if not df.empty:
    df.to_csv(output_file, index=False)
    print(f"✅ Text extracted and saved to {output_file} with {len(df)} entries")
else:
    print("⚠️ No data extracted. Check image paths and OCR settings.")

# Preview the first few entries
if not df.empty:
    print(df.head())