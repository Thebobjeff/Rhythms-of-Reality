import os
import csv
import lyricsgenius
from dotenv import load_dotenv
import runpy
import time
from tqdm import tqdm

Batch_Size = 50

# 1. Setup Environment
load_dotenv()
token = os.getenv("client_access_token")

# Initialize Genius with optimized settings
genius = lyricsgenius.Genius(token)
genius.verbose = False 
genius.remove_section_headers = True
genius.skip_non_songs = True 
 
input_file = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\hot100_Consolidated_Genres.csv'
output_file = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\finalDataset.csv'

def fetch_lyrics(row):
    """
    Fetches and formats lyrics for a single row.
    """
    artist = row.get('Artist', '')
    song_title = row.get('Song', '')
    
    # Clean up song titles
    clean_song = song_title.split(' (From ')[0].split(' - ')[0].strip()
    
    try:
        # Search using explicit title and artist
        result = genius.search_song(clean_song, artist)
        
        if result and result.lyrics:
            clean_lyrics = result.lyrics.replace("\n", " | ")
        else:
            clean_lyrics = "Lyrics Not Found"
            
    except Exception as e:
        clean_lyrics = f"Error: {str(e)}"
    
    new_row = row.copy()
    new_row['Lyrics'] = clean_lyrics
    return new_row

# Helper function to create batches
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def main():
    # 1. Read the file
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['Lyrics']
        rows = list(reader)

    print(f"Starting sequential lyrics fetch for {len(rows)} songs in batches of 50...")

    all_results = []
    
    # 2. Process in batches
    # We use tqdm to wrap the batching process
    batches = list(chunker(rows, Batch_Size))
    
    for batch in tqdm(batches, desc="Processing Batches"):
        for row in batch:
            # Process one by one sequentially
            result = fetch_lyrics(row)
            all_results.append(result)
        
        # Sleep for 1 second between batches
        time.sleep(1)

    # 3. Write the results to the new CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Successfully created: {output_file}")

if __name__ == "__main__":
    main()

# print("Running lyricsToVectors.py...")
# runpy.run_path('C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\funtions\\lyricsToVectors.py')