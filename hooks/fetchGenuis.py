import os
import csv
import lyricsgenius
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# 1. Setup Environment
load_dotenv()
token = os.getenv("client_access_token")

# Initialize Genius with optimized settings
genius = lyricsgenius.Genius(token)
genius.verbose = False 
genius.remove_section_headers = True
genius.skip_non_songs = True  # Prevents downloading tracklists or liner notes

input_file = 'test.csv'
output_file = 'updateTestFile1.csv'

def fetch_lyrics_task(row):
    """
    Worker function to process a single row.
    Accesses columns by name based on your CSV header.
    """
    artist = row['Artist']
    song_title = row['Song']
    
    # Clean up song titles that have movie info like (From "Young Guns II")
    # This helps Genius find the song more accurately
    clean_song = song_title.split(' (From ')[0]
    
    try:
        # Searching by (song, artist) is faster and more accurate than a single string
        result = genius.search_song(clean_song, artist)
        lyrics = result.lyrics if result else "Lyrics Not Found"
    except Exception as e:
        lyrics = "Song Not Found Bucko"
        # lyrics = f"Error: {str(e)}"
    
    # Replace newlines with a pipe and spaces
    clean_lyrics = lyrics.replace("\n", " | ")


    # Return the original data plus the new lyrics column
    return {**row, 'Lyrics': clean_lyrics}

def main():
    # 1. Read the file using DictReader
    # DictReader AUTOMATICALLY skips the header and uses it for keys
    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # We grab the fieldnames (headers) and add 'Lyrics' to the list
        fieldnames = reader.fieldnames + ['Lyrics']
        
        # Convert to a list so ThreadPoolExecutor can process it
        rows = list(reader)

    print(f"Starting lyrics fetch for {len(rows)} songs...")

    # 2. Process in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        final_results = list(executor.map(fetch_lyrics_task, rows))

    # 3. Write results
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_results)

    print(f"Successfully created: {output_file}")
    
if __name__ == "__main__":
    main()