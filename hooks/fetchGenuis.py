import os
import csv
import lyricsgenius
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import runpy

# 1. Setup Environment
load_dotenv()
token = os.getenv("client_access_token")

# Initialize Genius with optimized settings
genius = lyricsgenius.Genius(token)
genius.verbose = False 
genius.remove_section_headers = True
genius.skip_non_songs = True 

input_file = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\helperTest.csv'
# input_file = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\hot100_Consolidated_Genres.csv'
output_file = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\finalDataset.csv'

def fetch_lyrics_task(row):
    """
    Processes a single row, fetches lyrics, and formats them.
    """
    # Point 2: Accessing Title and Artist separately
    artist = row.get('Artist', '')
    song_title = row.get('Song', '')
    
    # Clean up song titles that have extra info like (From "Young Guns II")
    clean_song = song_title.split(' (From ')[0].split(' - ')[0].strip()
    
    try:
        # Search using explicit title and artist
        result = genius.search_song(clean_song, artist)
        
        if result and result.lyrics:
            # Point 3: Replace newlines with a pipe separator
            clean_lyrics = result.lyrics.replace("\n", " | ")
        else:
            clean_lyrics = "Lyrics Not Found"
            
    except Exception as e:
        clean_lyrics = f"Error: {str(e)}"
    
    # Return a new dictionary with the original data plus the Lyrics
    new_row = row.copy()
    new_row['Lyrics'] = clean_lyrics
    return new_row

def main():
    # 1. Read the file
    # DictReader automatically consumes the header row and uses it as keys
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['Lyrics']
        rows = list(reader)

    print(f"Starting lyrics fetch for {len(rows)} songs...")

    # 2. Process in parallel
    # Using 4-8 workers to stay within Genius API limits
    with ThreadPoolExecutor(max_workers=8) as executor:
        final_results = list(executor.map(fetch_lyrics_task, rows))

    # 3. Write the results to the new CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)

    print(f"Successfully created: {output_file}")

if __name__ == "__main__":
    main()


# print("Running lyricsToVectors.py...")
# runpy.run_path('C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\funtions\\lyricsToVectors.py')