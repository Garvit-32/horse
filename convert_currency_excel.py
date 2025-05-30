import pandas as pd
import re
# import ace_tools as tools

df = pd.read_excel("/home/ubuntu/workspace/all_sales_mapped.xlsx")

price_pattern = re.compile(r"([€£]|[A-Z]{1,3}\$?)?\s*([\d,\.]+)")

def split_price(cell):
    if pd.isna(cell):
        return None, None
    s = str(cell).strip()

    # guinea
    if "gns" in s.lower():
        amount = float(re.sub(r"[^0-9.]", "", s))
        return "GNS", amount

    m = price_pattern.match(s)
    if not m:
        return None, None

    symbol, number = m.groups()
    value = float(number.replace(",", "")) if number else None

    if not symbol:
        iso = "USD"
    elif symbol.endswith("$") and len(symbol) > 1:  # A$, NZ$, C$
        iso = {"A$": "AUD", "NZ$": "NZD", "C$": "CAD"}.get(symbol, symbol.rstrip("$"))
    else:
        iso = {"€": "EUR", "£": "GBP", "$": "USD"}.get(symbol, symbol)

    return iso, value

df[["Currency", "Amount"]] = list(map(list, df["Price"].apply(split_price)))

fx = {
    "USD": 1.0,
    "EUR": 1.09,
    "GBP": 1.27,
    "AUD": 0.66,
    "NZD": 0.61,
    "CAD": 0.74,
    "AED": 0.27,
    "JPY": 0.0064,
    "GNS": 1.27 * 1.05,
}


df["Price_USD"] = df.apply(lambda r: r.Amount * fx.get(r.Currency) if pd.notna(r.Amount) else None, axis=1)

out_file = "/home/ubuntu/workspace/all_sales_usd.xlsx"
df.to_excel(out_file, index=False)
print("Saved to", out_file)
# tools.display_dataframe_to_user("Parsed prices (first 20 rows)", df.head(20))
