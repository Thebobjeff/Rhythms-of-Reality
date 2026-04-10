import os
import csv
import lyricsgenius
from dotenv import load_dotenv
import time

# 1. Setup Environment
load_dotenv()
token = os.getenv("client_access_token")
genius = lyricsgenius.Genius(token)

# Optional: Genius settings to make it faster/cleaner
genius.verbose = False 
genius.remove_section_headers = True 

input_file = 'hot100.csv'
output_file = 'billboard_with.csv'

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    
    # We use a standard reader, but we have to handle your "split-by-comma" data manually
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        if not row:
            continue
            
        # BASED ON YOUR DATA FORMAT:
        # Index 0: Year
        # Last 2 indices: Peak Position and Date
        # Everything in between: Artist and Song parts
        
        year = row[0]
        peak = row[-2]
        date = row[-1]
        
        # We need to distinguish Artist from Song. 
        # Since your data is fragmented, this is a "best guess" logic:
        # We'll take everything between the Year and the Peak and try to search Genius.
        # NOTE: If your CSV is properly formatted as Year, Artist, Song, Peak, Date, use indices row[1] and row[2]
        
        # Reassembling fragmented names (joining middle parts with spaces)
        full_line_text = " ".join(row[1:-2]) 
        
        print(f"Searching for: {full_line_text}...")
        
        lyrics = "Lyrics Not Found"
        try:
            # search_song works best when you pass the whole string or (artist, title)
            song_result = genius.search_song(full_line_text)
            if song_result:
                lyrics = song_result.lyrics
        except Exception as e:
            print(f"Error fetching {full_line_text}: {e}")
            lyrics = "Error during fetch"

        # Write original data + the new lyrics column
        writer.writerow(row + [lyrics])
        
        # Small delay to avoid hitting Genius rate limits too hard
        time.sleep(1)

print(f"Done! Created {output_file}")