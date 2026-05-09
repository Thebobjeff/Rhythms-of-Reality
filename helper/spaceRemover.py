import csv
import re
import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent


# --- Configuration ---
csv.field_size_limit(sys.maxsize)

# Regex pattern to collapse multiple pipes into one
CLEAN_PIPE_PATTERN = re.compile(r'\|+')

def format_lyrics(text):
    """Replaces spaces/newlines with pipes and cleans up consecutive pipes."""
    if not text:
        return ""
    # Replace newlines and spaces with pipes
    cleaned = text.replace('\n', '|')
    # Collapse multiple pipes (|||) into one (|)
    return CLEAN_PIPE_PATTERN.sub('|', cleaned).strip('|')

def fix_lyrics_errors(file1, file2, outfile_path):
    error_string = "Error: "
    
    if not file2.exists():
        print(f"Error: {file2} not found.")
        return

    # Step 1: Load File 2 into a lookup dictionary
    # Key: (Year, Artist, Song) -> Value: Lyrics
    lookup_table = {}
    print(f"Reading source lyrics from {file2.name}...")
    
    with open(file2, mode='r', encoding='utf-8') as f2:
        reader2 = csv.reader(f2)
        next(reader2, None)  # Skip header
        for row in reader2:
            # File 2 Structure: 0:Year, 1:Artist, 2:Song, 3:Peak, 4:FullYear, 5:Lyrics
            if len(row) >= 6:
                year = row[0].strip()
                artist = row[1].strip().lower()
                song = row[2].strip().lower()
                lyrics = row[5]
                
                key = (year, artist, song)
                lookup_table[key] = lyrics

    # Step 2: Process File 1 and apply fixes
    print(f"Processing {file1.name} and fixing errors...")
    rows_fixed = 0
    total_rows = 0

    with open(file1, mode='r', encoding='utf-8') as f1, \
         open(outfile_path, mode='w', encoding='utf-8', newline='') as fout:
        
        reader1 = csv.reader(f1)
        writer = csv.writer(fout)
        
        # Handle Header for File 1
        header = next(reader1, None)
        if header:
            writer.writerow(header)

        for row in reader1:
            # File 1 Structure: 0:Year, 1:Artist, 2:Song, 3:Peak, 4:FullYear, 5:Genre, 6:Lyrics
            if len(row) >= 7:
                total_rows += 1
                year = row[0].strip()
                artist = row[1].strip().lower()
                song = row[2].strip().lower()
                current_lyrics = row[6]

                # If the error string is found in the lyrics column
                if error_string in current_lyrics:
                    key = (year, artist, song)
                    if key in lookup_table:
                        # Replace with lyrics from File 2
                        row[6] = format_lyrics(lookup_table[key])
                        rows_fixed += 1
                    else:
                        print(f"  [No Match Found] {row[1]} - {row[2]} ({row[0]})")
                        # Even if no match, format the existing error string or partial text
                        row[6] = format_lyrics(current_lyrics)
                else:
                    # If no error, just ensure the lyrics are formatted with pipes correctly
                    row[6] = format_lyrics(current_lyrics)
            
            writer.writerow(row)

    print("-" * 30)
    print(f"Done! Processed {total_rows} songs.")
    print(f"Fixed {rows_fixed} songs that had timeout errors.")
    print(f"Saved to {outfile_path}")

# --- EXECUTION ---

if __name__ == "__main__":
    

    # File paths based on your project structure
    infile1 = project_root / "data" / "CSV" / "finalDataset.CSV" 
    infile2 = project_root / "data" / "CSV" / "billboard_with.csv"
    outfile = project_root / "data" / "CSV" / "cleaned_dataset_Final_1.1.csv"

    fix_lyrics_errors(infile1, infile2, outfile)