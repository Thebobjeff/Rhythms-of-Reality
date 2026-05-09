import pandas as pd
from pathlib import Path

# --- 1. FILE PATHS ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

input_path  = project_root / "data" / "CSV" / "finalDatasetEmbeddings_2.0.csv"
output_path = project_root / "data" / "CSV" / "hipHopDataset.csv"

# --- 2. LOAD DATA ---
print("Loading dataset...")
df = pd.read_csv(input_path)
print(f"  Total songs loaded: {len(df)}")

# --- 3. INSPECT UNIQUE GENRES (helpful for confirming exact label) ---
print("\nGenres found in dataset:")
for genre, count in df["Genre"].value_counts().items():
    print(f"  {genre:<30} {count} songs")

# --- 4. FILTER HIP-HOP ---
hiphop_df = df[df["Genre"] == "Hip-Hop/Rap"].reset_index(drop=True)
print(f"\nHip-Hop/Rap songs found: {len(hiphop_df)}")

# --- 5. YEAR COVERAGE ---
print("\nSongs per year:")
for year, count in sorted(hiphop_df.groupby("Year").size().items()):
    print(f"  {year}: {count} songs")

# --- 6. SAVE ---
hiphop_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")