import pandas as pd
import os
from sentence_transformers import SentenceTransformer

def process_lyrics_to_vectors(input_path, output_path, column_name='Lyrics'):
    """
    Loads a CSV, generates embeddings for a specific column, 
    and saves the result to a new CSV.
    """
    # 1. Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 2. Load the model once
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Load the data
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 4. Generate Embeddings (Batching is much faster than .apply)
    print("Generating embeddings... (this may take a moment)")
    # We convert the column to a list and pass it to encode()
    embeddings = model.encode(df[column_name].tolist())
    
    # Assign the list of vectors back to the dataframe
    df['embeddings'] = embeddings.tolist()
    
    # 5. Save the data
    df.to_csv(output_path, index=False)
    print(f"Success! Saved to {output_path}")

# --- HOW TO USE IT ---
input_file = r'C:\Users\devon\Documents\GitHub\lang Chain\Test Project- git\data\CSV\finalDataset.csv'
output_file = r'C:\Users\devon\Documents\GitHub\lang Chain\Test Project- git\data\CSV\finalDatasetEmbeddings.csv'

process_lyrics_to_vectors(input_file, output_file)