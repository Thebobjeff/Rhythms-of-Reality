import os
import csv
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

input_file  = 'hooks/updateTestFile1.csv'
output_file = 'hooks/finalDataset.csv'

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


def main():
    # 1. Read input CSV
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['Genre']
        rows       = list(reader)

    unique_artists = list(set(r['Artist'] for r in rows))
    print(f"Found {len(unique_artists)} unique artists across {len(rows)} songs")
    print(f"Processing in batches of {BATCH_SIZE} via Groq...\n")

    # 2. Classify all unique artists in batches
    genre_map = {}
    batches   = [unique_artists[i:i+BATCH_SIZE] for i in range(0, len(unique_artists), BATCH_SIZE)]

    for i, batch in enumerate(batches, 1):
        print(f"Batch {i}/{len(batches)} ({len(batch)} artists)...")
        result = classify_artists_batch(batch)
        genre_map.update(result)
        print(f"  Sample: {list(result.items())[:2]}")
        time.sleep(0.3)  # Small pause

    # 3. Map genres back onto every row
    results = []
    for row in rows:
        genre = genre_map.get(row['Artist'], 'Unknown')
        results.append({**row, 'Genre': genre})

    # 4. Write final CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 5. Print genre breakdown summary
    from collections import Counter
    genre_counts = Counter(r['Genre'] for r in results)
    print(f"\n✅ Created: {output_file}")
    print(f"\nGenre breakdown:")
    for genre, count in genre_counts.most_common():
        bar = "█" * (count // 20)
        print(f"  {genre:<22} {count:>4} songs  {bar}")


if __name__ == "__main__":
    main()