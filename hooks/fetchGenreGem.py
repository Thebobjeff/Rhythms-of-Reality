import pandas as pd
import os
import json
import time
from dotenv import load_dotenv
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Changed imports from Groq to Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# gemmini can handle larger batches, but we'll keep it small to ensure we don't hit rate limits or safety filters pls lower the risk of a whole batch failing 
BATCH_SIZE = 50

GENRE_CATEGORIES = [
    "Pop", "Hip-Hop/Rap", "R&B/Soul", "Rock", "Country",
    "Dance/Electronic", "Latin", "Alternative", "Folk/Acoustic",
    "Gospel/Christian", "Metal/Hard Rock", "Reggae", "Jazz/Blues", "Unknown"
]

# Initialize Gemini LLM instead of Groq
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview", # Or "gemini-1.5-pro"
    temperature=0,
    thinking_level="low",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

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
        # This will tell you EXACTLY why it failed (Rate limit, Safety, or JSON error)
        print(f"  Error in batch: {e}") 
        return {a: "Unknown" for a in artists}

# 1. SET THE FILE PATH HERE
file_path = "hot100.csv" 

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
# df_filtered = df[1995 >= 1990].copy()

df_filtered = df[df['Year'] >= 1990].copy()

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

# --- GEMINI INTEGRATION ---

unique_artists = top_100_final['Artist'].unique().tolist()
artist_genre_map = {}

print(f"--- Classifying {len(unique_artists)} unique artists using Gemini ---")

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
top_100_final.to_csv('C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\hot100_Gemini.csv', index=False)

print("\nFile successfully created: hot100Gemini.csv")