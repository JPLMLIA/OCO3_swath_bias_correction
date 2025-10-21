#!/usr/bin/env python3
"""
Image Labeler for OCO-3 Swath Bias Correction

This script allows you to label images manually using a GUI.
"""




import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import re
import glob
import csv
import cv2
import random

# Configure the folder path - modify this for your setup
folder_path = os.getenv('OCO3_SAM_IMAGES_DIR', './data/sam_images/')

# Find all PNG files in the folder using glob
image_files = glob.glob(os.path.join(folder_path, "SAM*.png"))

# randomly subset images
# image_files = random.sample(image_files, 500)


# Initialize dictionary to hold labels
labels_dict = {}

print("Press '1' for yes/true, '0' for no/false, and 'q' to quit.")

for img_file in image_files:
    # Extract image identifier from the filename
    base_name = os.path.basename(img_file)
    match = re.search(r'SAM_(.*)_xco2\.png', base_name)
    if match:
        identifier = match.group(1)
    else:
        identifier = os.path.splitext(base_name)[0]

    # Read and display the image
    img = cv2.imread(img_file)
    if img is None:
        print(f"Could not load image: {img_file}")
        continue

    cv2.imshow("Image", img)
    key = cv2.waitKey(0) & 0xFF  # Wait for key press

    # Check the pressed key
    if key == ord('q'):
        print("Exiting labeling process.")
        break
    elif key == ord('1'):
        labels_dict[identifier] = 1
    elif key == ord('0'):
        labels_dict[identifier] = 0
    elif key == ord('2'):
        labels_dict[identifier] = 2
    else:
        print("Invalid key pressed. Skipping this image.")
        continue

    cv2.destroyAllWindows()  # Close the image window after key press

# Save the dictionary to a CSV file
csv_file = "Swath_Bias_labels2.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["identifier", "label"])
    for key, value in labels_dict.items():
        writer.writerow([key, value])

print(f"Labels saved to {csv_file}")
cv2.destroyAllWindows()