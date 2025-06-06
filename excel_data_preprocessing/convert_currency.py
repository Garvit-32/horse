import re
import pandas as pd
from datetime import datetime

# === 1) PARAMETERS ===
INPUT_PATH = "C:/Users/kapri/Desktop/Ayush-WORKSPACE/horse_project/currency_sym.xlsx"
OUTPUT_PATH = "C:/Users/kapri/Desktop/Ayush-WORKSPACE/horse_project/all_sales_mapped_usd_from_symbols.xlsx"

# === 2) READ THE EXCEL ===
df = pd.read_excel(INPUT_PATH)

# === 3) HELPER FUNCTION: EXTRACT RAW SYMBOL + NUMERIC PART ===


def extract_raw_symbol_and_amount(text):
    if pd.isna(text):
        return None, None

    s = str(text).strip()
    if s == "":
        return None, None

    # Capture any run of non-digits at the very start: ^(\D+)
    match_pref = re.match(r"^(\D+)", s)
    prefix = match_pref.group(1).strip() if match_pref else ""

    # Capture any run of non-digits at the very end: (\D+)$
    match_suf = re.search(r"(\D+)$", s)
    suffix = match_suf.group(1).strip() if match_suf else ""

    # Decide the "raw symbol": prefix if it exists, else suffix
    raw_symbol = prefix if prefix != "" else suffix

    # Extract digits only, then convert to float. If empty, return None.
    digits_only = re.sub(r"[^\d]", "", s)
    price_val = float(digits_only) if digits_only != "" else None

    return raw_symbol, price_val


# === 4) APPLY extract_raw_symbol_and_amount TO EACH ROW ===
df[['RawSymbol', 'PriceNumeric']] = df['PriceWithSymbol'].apply(
    lambda x: pd.Series(extract_raw_symbol_and_amount(x))
)

print("\nAfter extracting RawSymbol and PriceNumeric (first 10 rows):")
print(df[['PriceWithSymbol', 'RawSymbol', 'PriceNumeric']].head(10))


# === 5) MAP RAW SYMBOL → ISO CODE ===
RAW_TO_ISO = {
    '€':   'EUR',
    '£':   'GBP',
    '$':   'USD',
    'EUR': 'EUR',
    'GBP': 'GBP',
    'USD': 'USD',
    'AUD': 'AUD',
    'NZD': 'NZD',
    'CAD': 'CAD',
    'CHF': 'CHF',
    'GNS': 'GBP',
}


def map_raw_to_iso(raw):
    if raw is None or str(raw).strip() == "":
        return None
    key = str(raw).strip().upper()
    return RAW_TO_ISO.get(key)


df['CurrencyDetected'] = df['RawSymbol'].apply(map_raw_to_iso)

print("\nAfter mapping RawSymbol → CurrencyDetected (first 10 rows):")
print(df[['PriceWithSymbol', 'RawSymbol',
      'CurrencyDetected', 'PriceNumeric']].head(10))


# === 6) COUNT UNIQUE RAW SYMBOLS & THEIR FREQUENCIES ===
raw_counts = (
    df['RawSymbol']
    .dropna()
    .astype(str)
    .str.strip()
    .replace("", pd.NA)           # Treat empty string as missing
    .value_counts(dropna=True)    # Count how many times each symbol appears
)

print("\nRawSymbol counts (symbol → number of rows):")
for sym, cnt in raw_counts.items():
    print(f"  • '{sym}': {cnt}")

no_symbol_count = df['RawSymbol'].isna().sum() \
    + (df['RawSymbol'].astype(str).str.strip() == "").sum()
print(f"Number of rows with no detected RawSymbol: {no_symbol_count}")


# === 7) CHECK FOR EMPTY PriceWithSymbol CELLS ===
empty_price_count = (
    df['PriceWithSymbol'].isna().sum()
    + (df['PriceWithSymbol'].astype(str).str.strip() == "").sum()
)
print(
    f"\nNumber of rows where PriceWithSymbol is empty/blank: {empty_price_count}")

if empty_price_count > 0:
    print("\nSample rows where PriceWithSymbol is empty:")
    example_empty = df[df['PriceWithSymbol'].astype(str).str.strip() == ""]
    print(example_empty.head(5))


# === 8) CHECK FOR "ONLY NUMBERS, NO SYMBOL" ROWS ===
empty_mask = (
    df['PriceWithSymbol'].isna()
    |
    (df['PriceWithSymbol'].astype(str).str.strip() == "")
)

empty_price_count = empty_mask.sum()
print(
    f"Number of rows where PriceWithSymbol is empty/blank: {empty_price_count}")

if empty_price_count > 0:
    print("\nSample rows where PriceWithSymbol is empty:")
    print(df.loc[empty_mask, ['Title', 'Price',
          'PriceWithSymbol', 'Purchaser', 'PDF_Path']].head(5))

# ------------------------------------------------------------------
# 8-B)  LIST UNIQUE PDF_Path VALUES FOR "ONLY-NUMBER" ROWS
# ------------------------------------------------------------------
# --------------------------------------------------------------
# Unique "sale-house" (2nd folder) counts for *only-number* rows
# --------------------------------------------------------------
only_number_mask = (
    (df['RawSymbol'].astype(str).str.strip() == "")
    & (df['PriceNumeric'].notna())
)

salehouse_counts = (
    df.loc[only_number_mask, 'PDF_Path']
      .dropna()
      .apply(lambda p: str(p).split('/')[1] if '/' in str(p) else None)
      .dropna()
      .value_counts()
)

print("\nSale-house counts (based on 2nd directory in PDF_Path):")
for sh, cnt in salehouse_counts.items():
    print(f"  • {cnt:5d} × {sh}")


# === 9) MAP "ONLY-NUMBER" ROWS VIA PDF_Path SALEHOUSE → Currency ===
def get_currency_from_salehouse(pdf_path):
    if pd.isna(pdf_path):
        return None

    # Extract sale-house from PDF_Path (2nd folder)
    parts = str(pdf_path).split('/')
    if len(parts) < 2:
        return None

    salehouse = parts[1].lower()

    # Map sale-house to currency
    if salehouse == 'fastiptondata':
        return 'USD'
    elif salehouse == 'newzealand':
        return 'NZD'
    return None


# Update CurrencyDetected for rows with only numbers
only_number_mask = (
    (df['RawSymbol'].astype(str).str.strip() == "")
    & (df['PriceNumeric'].notna())
)

# Apply the mapping for rows with only numbers
df.loc[only_number_mask, 'CurrencyDetected'] = df.loc[only_number_mask,
                                                      'PDF_Path'].apply(get_currency_from_salehouse)

# Print summary of currency assignments
print("\nCurrency assignments for rows with only numbers:")
currency_counts = df.loc[only_number_mask, 'CurrencyDetected'].value_counts()
for currency, count in currency_counts.items():
    print(f"  • {count:5d} rows assigned to {currency}")


# === 10) FX RATES & CONVERSION TO USD ===
FX_TO_USD = {
    'USD': 1.00,    # Base currency
    'EUR': 1.15,    # Updated from 1.09
    'GBP': 1.36,    # Updated from 1.27
    'AUD': 0.65,    # Updated from 0.66
    'NZD': 0.61,    # Updated from 0.61
    'CAD': 0.73,    # Updated from 0.75
    'CHF': 1.22,    # Updated from 1.10
}

# Print conversion rates with date
current_date = datetime.now().strftime("%Y-%m-%d")
print(f"\nCurrency Conversion Rates to USD (as of {current_date}):")
for currency, rate in FX_TO_USD.items():
    print(f"  • 1 {currency} = {rate:.2f} USD")


def convert_to_usd(amount, ccy_code):
    if amount is None or pd.isna(amount):
        return None
    if not isinstance(ccy_code, str):
        return None
    iso = ccy_code.strip().upper()
    rate = FX_TO_USD.get(iso)
    if rate is None:
        return None
    return amount * rate


# Convert to USD
df['USD'] = df.apply(
    lambda row: convert_to_usd(row['PriceNumeric'], row['CurrencyDetected']),
    axis=1
)

# Print conversion summary
print("\nCurrency Conversion Summary:")
currency_summary = df.groupby('CurrencyDetected').agg({
    'PriceNumeric': ['count', 'sum'],
    'USD': 'sum'
}).round(2)

print(currency_summary)

# === 11) SAVE TO EXCEL ===
# Add a timestamp to the output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = f"C:/Users/kapri/Desktop/Ayush-WORKSPACE/horse_project/all_sales_mapped_usd_{timestamp}.xlsx"

# Save with all relevant columns
columns_to_save = [
    'Title', 'Price', 'PriceWithSymbol', 'Purchaser', 'PDF_Path',
    'RawSymbol', 'PriceNumeric', 'CurrencyDetected', 'USD'
]

df[columns_to_save].to_excel(OUTPUT_PATH, index=False)
print(f"\n✅ Saved converted file to:\n   {OUTPUT_PATH}")

# Print final summary
print("\nFinal Summary:")
print(f"Total rows processed: {len(df)}")
print(f"Rows with USD conversion: {df['USD'].notna().sum()}")
print(f"Rows without USD conversion: {df['USD'].isna().sum()}")
