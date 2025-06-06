import pandas as pd
from dateutil import parser

# Read the CSV
df = pd.read_csv('/home/ubuntu/workspace/all_data.csv')

def parse_dob(dob):
    dob_str = str(dob)
    if ' ' in dob_str:  # likely a full date
        try:
            return parser.parse(dob_str)
        except Exception:
            pass
    # Try to treat as year
    try:
        year = int(dob_str[:4])
        return pd.Timestamp(year=year, month=1, day=1)
    except Exception:
        return pd.NaT

# Parse dob
df['dob_parsed'] = df['dob'].apply(parse_dob)
df = df.dropna(subset=['dob_parsed'])

# Sort by dob_parsed (oldest to newest)
df = df.sort_values('dob_parsed')
print(df.head())
# Split
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].drop(columns=['dob_parsed'])
test_df = df.iloc[split_idx:].drop(columns=['dob_parsed'])

# Save
train_df.to_csv('train_data_split.csv', index=False)
test_df.to_csv('test_data_split.csv', index=False)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
