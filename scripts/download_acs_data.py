from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# PA Open Data: Income & Poverty, ACS 5-Year Estimates, County
URL = "https://data.pa.gov/resource/6npu-erdk.csv?$limit=50000"

RAW_OUTPUT = RAW_DIR / "pa_income_poverty_acs.csv"
INSPECTION_OUTPUT = PROCESSED_DIR / "acs_inspection_output.txt"

print("Downloading ACS income and poverty data...")
df = pd.read_csv(URL)

df.to_csv(RAW_OUTPUT, index=False)

with open(INSPECTION_OUTPUT, "w", encoding="utf-8") as f:
    f.write("ACS Income and Poverty Dataset Inspection\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Shape: {df.shape}\n\n")

    f.write("Columns:\n")
    for col in df.columns:
        f.write(f"- {col}\n")

    f.write("\nFirst 10 Rows:\n")
    f.write(df.head(10).to_string(index=False))

    f.write("\n\nMissing Values:\n")
    f.write(df.isna().sum().to_string())

print("Download complete.")
print(f"Raw data saved to: {RAW_OUTPUT}")
print(f"Inspection saved to: {INSPECTION_OUTPUT}")