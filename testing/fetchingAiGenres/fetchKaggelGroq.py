import pandas as pd
import os
import json
import time
from dotenv import load_dotenv
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


BATCH_SIZE = 50

GENRE_CATEGORIES = [
    "Pop", "Hip-Hop/Rap", "R&B/Soul", "Rock", "Country",
    "Dance/Electronic", "Latin", "Alternative", "Folk/Acoustic",
    "Gospel/Christian", "Metal/Hard Rock", "Reggae", "Jazz/Blues", "Unknown"
]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt_template = ChatPromptTemplate.from_template("{request}")
parser = StrOutputParser()
chain  = prompt_template | llm | parser


def classify_artists_batch(artists: list) -> dict:
    """
    Sends a batch of artist names to Groq and returns
    a dict mapping each artist name to a genre string.
    """
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
        raw_text = chain.invoke({"request": request}).strip()

        # Strip markdown fences if Groq adds them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        return json.loads(raw_text.strip())

    except Exception as e:
        print(f"  Warning: batch failed ({e}), marking as Unknown")
        return {a: "Unknown" for a in artists}


# 1. SET THE FILE PATH HERE
# This tells kagglehub which specific file inside the 'ludmin/billboard' dataset to load
file_path = "hot100.csv" 

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ludmin/billboard",
  file_path,
)
 
# 2. Data Cleaning (Per your provided logic)
# Convert '-' to 0 and change types to integers
df['Weeks in Charts'] = df['Weeks in Charts'].replace('-', 0).astype(int)
df['Last Week'] = df['Last Week'].replace('-', 0).astype(int)

# Convert Date to datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# 3. Filter and Extract Year
df['Year'] = df['Date'].dt.year
df_filtered = df[df['Year'] >= 1990].copy()

# 4. Aggregate to find the best performance of each song per year
# We group by Year, Artist, and Song to find the highest rank (minimum value)
yearly_stats = df_filtered.groupby(['Year', 'Artist', 'Song']).agg({
    'Rank': 'min',
    'Date': 'min'
}).reset_index()

# 5. Rename for the final CSV format
yearly_stats.rename(columns={
    'Rank': 'Peak_Pos_That_Year', 
    'Date': 'Full_Release_Year'
}, inplace=True)

# 6. Sort by Year and Rank, then take the Top 100 for each year
top_100_final = yearly_stats.sort_values(['Year', 'Peak_Pos_That_Year']).groupby('Year').head(100)

# --- GROQ INTEGRATION ---

# 1. Get unique artists to save on API calls
unique_artists = top_100_final['Artist'].unique().tolist()
artist_genre_map = {}

print(f"--- Classifying {len(unique_artists)} unique artists using Groq ---")

for i in range(0, len(unique_artists), BATCH_SIZE):
    batch = unique_artists[i : i + BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1}...")
    results = classify_artists_batch(batch)
    artist_genre_map.update(results)
    time.sleep(1) # Safety delay for rate limits

# 2. Map genres back to the dataframe (this is your 'sample field')
top_100_final['Genre'] = top_100_final['Artist'].map(artist_genre_map).fillna("Unknown")

# 3. Print Artist and Song Name before exporting
print("\n--- Exporting the following songs ---")
for index, row in top_100_final.iterrows():
    print(f"Year: {row['Year']} | Artist: {row['Artist']} | Song: {row['Song']} | Genre: {row['Genre']}")

# 7. Export to CSV
top_100_final.to_csv('hot100_Groq.csv', index=False)

print("\nFile successfully created: hot100Groq.csv")