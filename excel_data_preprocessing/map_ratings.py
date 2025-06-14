# read this excel C:\Users\kapri\Desktop\Ayush-WORKSPACE\horse_project\Report 05062025.xlsx
import pandas as pd
from difflib import SequenceMatcher
import re
from tqdm import tqdm
import numpy as np
import os
import pickle
from datetime import datetime


def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', '', str(text).lower())


def calculate_similarity(str1, str2):
    str1 = clean_text(str1)
    str2 = clean_text(str2)
    if len(str1) < 3 or len(str2) < 3:  # Skip very short strings
        return 0
    return SequenceMatcher(None, str1, str2).ratio()


def create_year_sire_dam_index(df):
    """Create a nested dictionary structure for faster lookups"""
    index = {}
    for _, row in df.iterrows():
        year = str(row['BirthYear'])
        sire = clean_text(row['sireName'])
        dam = clean_text(row['damName'])

        if year not in index:
            index[year] = {}
        if sire not in index[year]:
            index[year][sire] = {}
        index[year][sire][dam] = row['MaxPerformanceRating']
    return index


def process_chunk(chunk_df, index, similarity_threshold=0.75):
    """Process a chunk of data and return matches"""
    horse_ratings = {}
    total_matches = 0

    for _, row in chunk_df.iterrows():
        if pd.isna(row['dob']):
            continue

        year = row['year']
        sire_clean = row['sire_clean']
        dam_clean = row['dam_clean']

        # Skip if year not in index
        if year not in index:
            continue

        # Find best matching sire
        best_sire_match = None
        best_sire_similarity = 0

        for sire in index[year].keys():
            similarity = calculate_similarity(sire_clean, sire)
            if similarity > best_sire_similarity:
                best_sire_similarity = similarity
                best_sire_match = sire

        if best_sire_similarity < similarity_threshold:
            continue

        # Find best matching dam
        best_dam_match = None
        best_dam_similarity = 0

        for dam in index[year][best_sire_match].keys():
            similarity = calculate_similarity(dam_clean, dam)
            if similarity > best_dam_similarity:
                best_dam_similarity = similarity
                best_dam_match = dam

        if best_dam_similarity >= similarity_threshold:
            horse_ratings[row['lot_number']
                          ] = index[year][best_sire_match][best_dam_match]
            total_matches += 1

    return horse_ratings, total_matches


def save_split_parts(df, output_dir, num_parts=8):
    """Save split parts to files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_parts = np.array_split(df, num_parts)

    for i, part in enumerate(df_parts):
        output_file = os.path.join(output_dir, f'part_{i+1}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(part, f)
        print(f"Saved part {i+1} to {output_file}")


def load_split_parts(input_dir):
    """Load split parts from files"""
    parts = []
    for file in sorted(os.listdir(input_dir)):
        if file.startswith('part_') and file.endswith('.pkl'):
            with open(os.path.join(input_dir, file), 'rb') as f:
                parts.append(pickle.load(f))
    return parts


def main(debug=True):
    # Define paths
    base_dir = r"C:\Users\kapri\Desktop\Ayush-WORKSPACE\horse_project"
    split_dir = os.path.join(base_dir, "splitted_parts")
    debug_dir = os.path.join(base_dir, "debug_mappings")

    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    # Read df2 (the smaller dataset)
    print("Reading smaller dataset...")
    df2 = pd.read_csv(os.path.join(base_dir, "all_sale_data.csv"))

    # Check if split parts exist
    if not os.path.exists(split_dir) or not os.listdir(split_dir):
        print("Split parts not found. Creating them...")
        df = pd.read_excel(os.path.join(base_dir, "Report 05062025.xlsx"))
        save_split_parts(df, split_dir)

    # Load split parts
    print("Loading split parts...")
    df_parts = load_split_parts(split_dir)

    # Pre-process df2
    print("Pre-processing data...")
    df2['year'] = df2['dob'].astype(str).str[-4:]
    df2['sire_clean'] = df2['sire_name'].apply(clean_text)
    df2['dam_clean'] = df2['dam_1_name'].apply(clean_text)

    # Process each part of df
    all_ratings = {}
    total_matches = 0

    for i, df_part in enumerate(df_parts):
        print(f"\nProcessing part {i+1}/{len(df_parts)}...")

        # Create index for this part
        print("Creating index...")
        index = create_year_sire_dam_index(df_part)

        # Process df2 in chunks
        chunk_size = 1000
        for start_idx in tqdm(range(0, len(df2), chunk_size), desc=f"Processing chunks for part {i+1}"):
            chunk = df2.iloc[start_idx:start_idx + chunk_size]
            chunk_ratings, chunk_matches = process_chunk(chunk, index)

            all_ratings.update(chunk_ratings)
            total_matches += chunk_matches

            # Save chunk results for verification
            chunk_with_ratings = chunk.copy()
            chunk_with_ratings['MaxPerformanceRating'] = chunk_with_ratings['lot_number'].map(
                chunk_ratings)

            # Save to Excel with timestamp and chunk info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chunk_file = os.path.join(
                debug_dir, f'chunk_part{i+1}_start{start_idx}_{timestamp}.xlsx')
            chunk_with_ratings.to_excel(chunk_file, index=False)
            print(f"\nSaved chunk results to: {chunk_file}")
            print(f"Matches found in this chunk: {chunk_matches}")

            print(f"\nTotal matches found so far: {total_matches}")

    # Add the ratings to df2
    print("\nAdding ratings to dataframe...")
    df2['MaxPerformanceRating'] = df2['lot_number'].map(all_ratings)

    # Save the updated dataframe
    print("Saving results...")
    output_file = os.path.join(base_dir, 'all_sale_data_with_ratings.csv')
    df2.to_csv(output_file, index=False)

    print(f"\nMapping complete. Results saved to '{output_file}'")
    print(f"Total matches found: {total_matches}")
    print(f"Match rate: {(total_matches/len(df2))*100:.2f}%")


if __name__ == "__main__":
    main(debug=True)  # Set to True to enable debug mode
