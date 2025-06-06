import os
import csv
import pandas as pd
import numpy as np
from dateutil import parser

def subfolder_path(path):
    # Split by '/' and get the third part (index 2)
    parts = path.split('/')
    return '/'.join(parts[2:-1])

# Set your data directory and input Excel file
data_dir = "/home/ubuntu/workspace/best_view_data"
input_excel = "/home/ubuntu/workspace/mapped_dob_title_usd_sorted.xlsx"
output_csv = "/home/ubuntu/workspace/all_data.csv"

# Read the Excel file
df = pd.read_excel(input_excel, usecols=["dob", "USD", "path"])

valid_entries = []

for idx, row in df.iterrows():
    folder_path = os.path.join(data_dir, str(subfolder_path((row['path']))))
    # breakpoint()
    if os.path.exists(folder_path):
        valid_entries.append([row['dob'], row['USD'], str(subfolder_path((row['path'])))])

# Remove entries with USD == 0 and rename USD to price
filtered_entries = [
    [row[0], row[1], row[2]]  # dob, USD, path
    for row in valid_entries if row[1] != 0
]

# Optionally, rename columns for training
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['dob', 'price', 'path'])  # header: USD -> price
    for row in filtered_entries:
        # row[1] is USD, which we now call price
        writer.writerow([row[0], row[1], row[2]])

print(f"Total valid entries (USD != 0) written to {output_csv}: {len(filtered_entries)}")







