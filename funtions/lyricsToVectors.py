import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
Batch_Size = 50

def process_lyrics_to_vectors(input_path, output_path, column_name='Lyrics'):
    """
    Loads a CSV, generates embeddings in batches of 50, 
    and saves the result to a new CSV.
    """
    # 1. Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Load the model
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Load the data
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check if the column exists
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in CSV.")
        return

    # 4. Generate Embeddings with batching
    # batch_size=50: Processes 50 songs at a time
    # show_progress_bar=True: Gives you a nice progress bar
    print("Generating embeddings (Batch size = 50)...")
    embeddings = model.encode(
        df[column_name].tolist(), 
        batch_size=Batch_Size, 
        show_progress_bar=True
    )
    
    # Assign the list of vectors back to the dataframe
    df['embeddings'] = embeddings.tolist()
    
    # 5. Save the data
    df.to_csv(output_path, index=False)
    print(f"Success! Saved to {output_path}")

# --- HOW TO USE IT ---
input_file = project_root / "data" / "CSV" / "cleaned_dataset_Final_1.1.csv"
output_file = project_root / "data" / "CSV" / "finalDatasetEmbeddings_1.csv"
 

process_lyrics_to_vectors(input_file, output_file)