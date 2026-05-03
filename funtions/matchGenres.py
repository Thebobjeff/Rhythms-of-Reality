import pandas as pd
import os
from pathlib import Path

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

# --- 1. SETTINGS & FILE PATHS ---
# Update these to your actual file names
openAi_Path = project_root / "data" / "CSV" / "hot100_OpenAI.csv"
gem_Path = project_root / "data" / "CSV" / "hot100_Gemini.csv"
groq_Path = project_root / "data" / "CSV" / "hot100_Groq.csv"
output_path = project_root / "data" / "CSV" / "hot100_Consolidated.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- 2. LOAD DATA ---
openAi_df = pd.read_csv(openAi_Path)
gem_df = pd.read_csv(gem_Path)
groq_df = pd.read_csv(groq_Path)

# --- 3. MERGE DATA ---
# We keep all columns from openAi_df and only pull the Genre from gem_df and groq_df to compare
# We match specifically on Artist and Song
merged = openAi_df.merge(
    gem_df[['Artist', 'Song', 'Genre']], on=['Artist', 'Song'], suffixes=('', '_Gemini')
).merge(
    groq_df[['Artist', 'Song', 'Genre']], on=['Artist', 'Song'], suffixes=('_OpenAI', '_Groq')
)

# After merging, your columns will be:
# 'Year', 'Artist', 'Song', 'Peak_Pos_That_Year', 'Full_Release_Year'
# 'Genre_OpenAI', 'Genre_Gemini', 'Genre_Groq'

# --- 4. Managed Democracy #FOR SUPER EARTH
def calculate_majority(row):
    openAi_Genres = str(row['Genre_OpenAI']).strip()
    gemini_Genres = str(row['Genre_Gemini']).strip()
    groq_Genres = str(row['Genre_Groq']).strip()
    
    # 1. If all match or 2/3 match
    if openAi_Genres == gemini_Genres:
        return openAi_Genres
    if openAi_Genres == groq_Genres:
        return openAi_Genres
    if gemini_Genres == groq_Genres:
        return gemini_Genres
    
    # 2. If all three are different
    return "Unknown"

# --- 5. APPLY LOGIC & CLEAN UP ---
print("Comparing genres and calculating majority...")

merged['Genre'] = merged.apply(calculate_majority, axis=1)

# Keep only the original columns in the final output
final_columns = ['Year', 'Artist', 'Song', 'Peak_Pos_That_Year', 'Full_Release_Year', 'Genre']
final_df = merged[final_columns]

# --- 6. SAVE RESULT ---
final_df.to_csv(output_path, index=False)

print(f"Success! Consolidated file saved to: {output_path}")

# Display a few results to verify
print("\nSample of consolidated results:")
print(final_df[['Artist', 'Song', 'Genre']].head(10))