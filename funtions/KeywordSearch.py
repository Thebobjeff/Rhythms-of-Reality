import pandas as pd
from pathlib import Path

# File Path set up
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent

# Different files being used
input_path = project_root / "data" / "CSV" / "finalDatasetEmbeddings_2.0.csv"
output_path = project_root / "data" / "keySearch" / "results.csv"


# KeyWord to search
keyword = "Step by step"

# Loading in data sheet
print("...loading Data Sheet")
df = pd.read_csv(input_path)
if(df.empty):
    print("Sheet Loaded Starting Process")
    print(f"Total Songs Loaded: {len(df)}")

# Searching for key word in lyrics column 
# na=False handles empty cells, case=False makes it case-insensitive
matched_songs = df[df['Lyrics'].str.contains(keyword, na=False, case=False)]
print(f"Case Matches: {len(matched_songs)}")

matched_songs.to_csv(output_path, index=False)
print(f"File Created: {output_path}")