import pandas as pd
import os
import json
import time
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter
# Changed imports from Google GenAI to OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent

load_dotenv()

# OpenAI batching is stable, keeping BATCH_SIZE at 50 as requested
BATCH_SIZE = 50

GENRE_CATEGORIES = [
    "Pop", "Hip-Hop/Rap", "R&B/Soul", "Rock", "Country",
    "Dance/Electronic", "Latin", "Alternative", "Folk/Acoustic",
    "Gospel/Christian", "Metal/Hard Rock", "Reggae", "Jazz/Blues", "Unknown"
]

# Initialize OpenAI LLM instead of Gemini
llm = ChatOpenAI(
    model="gpt-4o", # You can also use "gpt-4o-mini" for lower cost
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
# --- ADJUSTED AI PARTS END ---

prompt_template = ChatPromptTemplate.from_template("{request}")
parser = StrOutputParser()
chain = prompt_template | llm | parser | StrOutputParser()

def classify_artists_batch(artists: list) -> dict:
    artist_list = "\n".join(f"- {a}" for a in artists)
    categories  = ", ".join(GENRE_CATEGORIES)

    request = f"""You are a music genre classifier.
For each artist below, assign exactly ONE genre from this list: {categories}

Base your answer on the artist's PRIMARY genre.
For featured artists like "Drake Featuring Nicki Minaj", classify by the FIRST/MAIN artist only.
For artists known for R&B/Soul who also had pop crossover hits (e.g. Mariah Carey, Janet Jackson), classify as R&B/Soul
Return ONLY a valid JSON object with no extra text, no markdown, no explanation.
Format: {{"Artist Name": "Genre", "Artist Name 2": "Genre"}}

Artists to classify:
{artist_list}"""

    try:
        response = chain.invoke({"request": request}).strip()
        
        # More robust JSON cleaning
        clean_json = response
        if "```" in clean_json:
            clean_json = clean_json.split("```")[1]
            if clean_json.startswith("json"):
                clean_json = clean_json[4:]
        
        return json.loads(clean_json.strip())

    except Exception as e:
        print(f"  Error in batch: {e}") 
        return {a: "Unknown" for a in artists}

# 1. SET THE FILE PATH HERE
file_path = project_root / "data" / "CSV" / "hot100.csv" 

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ludmin/billboard",
  file_path,
)
 
# 2. Data Cleaning
# Convert '-' to 0 and change types to integers
df['Weeks in Charts'] = df['Weeks in Charts'].replace('-', 0).astype(int)
df['Last Week'] = df['Last Week'].replace('-', 0).astype(int)

# Convert Date to datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# 3. Filter and Extract Year
df['Year'] = df['Date'].dt.year

# Filtering Songs from 2025 during testing, change back to 1990 for final run
df_filtered = df[df['Year'] >= 2025].copy()

# 4. Aggregate
yearly_stats = df_filtered.groupby(['Year', 'Artist', 'Song']).agg({
    'Rank': 'min',
    'Date': 'min'
}).reset_index()

# 5. Rename
yearly_stats.rename(columns={
    'Rank': 'Peak_Pos_That_Year', 
    'Date': 'Full_Release_Year'
}, inplace=True)

# 6. Sort and Top 100
top_100_final = yearly_stats.sort_values(['Year', 'Peak_Pos_That_Year']).groupby('Year').head(100)

# --- OPENAI INTEGRATION ---

unique_artists = top_100_final['Artist'].unique().tolist()
artist_genre_map = {}

print(f"--- Classifying {len(unique_artists)} unique artists using OpenAI ---")

for i in range(0, len(unique_artists), BATCH_SIZE):
    batch = unique_artists[i : i + BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1}...")
    results = classify_artists_batch(batch)
    artist_genre_map.update(results)
    time.sleep(1) 

# 2. Map genres back
top_100_final['Genre'] = top_100_final['Artist'].map(artist_genre_map).fillna("Unknown")

# 3. Print Summary
print("\n--- Exporting the following songs ---")
for index, row in top_100_final.iterrows():
    print(f"Year: {row['Year']} | Artist: {row['Artist']} | Song: {row['Song']} | Genre: {row['Genre']}")

# 7. Export to CSV
top_100_final.to_csv(project_root / "data" / "CSV" / "hot100_OpenAI.csv", index=False)

print("\nFile successfully created: hot100_OpenAI.csv")