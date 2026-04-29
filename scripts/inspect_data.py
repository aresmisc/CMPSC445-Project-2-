from pathlib import Path
import pandas as pd

DATA_FILE = Path("data/raw/pa_county_health_2025.xlsx")
OUTPUT_FILE = Path("data/processed/inspection_output.txt")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Make sure the file is saved as "
        "data/raw/pa_county_health_2025.xlsx"
    )

xls = pd.ExcelFile(DATA_FILE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("Sheet Names:\n")
    for sheet in xls.sheet_names:
        f.write(f"- {sheet}\n")

    f.write("\n\nPreview of Each Sheet:\n")

    for sheet in xls.sheet_names:
        f.write(f"\n==================== {sheet} ====================\n")
        preview = pd.read_excel(DATA_FILE, sheet_name=sheet, header=None, nrows=12)
        f.write(preview.to_string(index=False, header=False))
        f.write("\n")

print("Inspection finished.")
print(f"Output saved to: {OUTPUT_FILE}")