import pandas as pd
import os

# Read the relevant columns
df_csv = pd.read_csv('/home/ubuntu/workspace/all_sale_data.csv', usecols=['dob', 'path'])
df_xlsx = pd.read_excel('/home/ubuntu/workspace/all_sales_mapped_usd_20250605_195823.xlsx', usecols=['PDF_Path', 'USD'])

# Extract the title from the path (third directory)
def extract_title(path):
    # Split by '/' and get the third part (index 2)
    parts = path.split('/')
    return parts[2] + '/' + parts[3] if len(parts) > 2 else None

df_csv['title'] = df_csv['path'].apply(extract_title)

# Build a mapping from extracted title to USD by searching for the substring in the Excel titles
title_to_usd = {}
unique_titles = df_csv['title'].dropna().unique()

for idx, row in df_xlsx.iterrows():
    excel_title = row['PDF_Path']
    if not isinstance(excel_title, str):
        continue
    for t in unique_titles:
        if isinstance(t, str) and t in excel_title:
            title_to_usd[t] = row['USD']

# Now map USD using the dictionary
df_csv['USD'] = df_csv['title'].map(title_to_usd)

# Remove rows where USD is NaN
result = df_csv[['dob', 'title', 'USD','path']].dropna(subset=['USD'])

# Get the total number of rows after dropping NaNs
print("Total length:", len(result))

# Optionally, display the first few rows
print(result.head())

# Sort by USD in descending order
result_sorted = result.sort_values(by='USD', ascending=True)

# Save the sorted result to an Excel file
result_sorted.to_excel('/home/ubuntu/workspace/mapped_dob_title_usd_sorted.xlsx', index=False)

