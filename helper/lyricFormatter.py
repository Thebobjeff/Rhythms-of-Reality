import re
import sys
from pathlib import Path

# --- Configuration ---
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
outfile = project_root / "data" / "testing" / "missingData.txt"

# Regex pattern to collapse multiple pipes into one
CLEAN_PIPE_PATTERN = re.compile(r'\|+')

def format_lyrics(text):
    """Replaces spaces/newlines with pipes and cleans up consecutive pipes."""
    if not text:
        return ""
    # Replace newlines and spaces with pipes
    cleaned = text.replace('\n', ' | ')
    # Collapse multiple pipes (|||) into one (|)
    return CLEAN_PIPE_PATTERN.sub("|", cleaned).strip("|")

def manual_lyrics_entry():
    print(f"--- Lyrics Entry Mode ---")
    print(f"Saving to: {outfile}")
    print("Instructions:")
    print("1. Paste your lyrics.")
    print("2. When finished with one song, type 'DONE' on a new line and press Enter.")
    print("3. Type 'QUIT' to exit the program.")
    print("-" * 30)

    while True:
        lines = []
        print("\n[Waiting for lyrics input...]")
        
        while True:
            line = sys.stdin.readline()
            
            # Check for commands
            stripped_line = line.strip().upper()
            if stripped_line == "DONE":
                break
            if stripped_line == "QUIT":
                print("Exiting...")
                return

            lines.append(line)

        # Join the lines into one block of text
        raw_text = "".join(lines)
        
        if raw_text.strip():
            # Format using your logic
            formatted_text = format_lyrics(raw_text)
            
            # Append to the file (Mode 'a' for append)
            with open(outfile, mode='a', encoding='utf-8') as f:
                # f.write(f"\"{formatted_text}\"\n")
                f.write(f"{formatted_text} \n")
                

            
            print(f"Successfully added {len(formatted_text)} characters (formatted) to the file.")
        else:
            print("No lyrics detected, skipping...")

if __name__ == "__main__":
    # Ensure the directory exists
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the loop
    manual_lyrics_entry()