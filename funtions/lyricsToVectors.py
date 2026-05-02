from sentence_transformers import SentenceTransformer
import pandas as pd
import os


# --- 1. SETTINGS & FILE PATHS ---
# Update these to your actual file names
testFile = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\finalDatasetSample.csv'
output_path = 'C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\finalDatasetSample_Embeddings.csv'


# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)


# --- 2. LOAD DATA ---
file = pd.read_csv(testFile)


#--- 3. GENERATE EMBEDDINGS ---
# The model creates a 384-dimensional vector. Each number represents a specific feature of the text's meaning.
# ```python
# # This is a truncated view of the actual 384-dimension array    -> with the .tolist() method, we convert the numpy array to a regular Python list, which is easier to store in a CSV file.
# array([ 0.0124, -0.4561,  0.8923, ..., -0.1102], dtype=float32) -> [0.0124, -0.4561, 0.8923, ..., -0.1102] 

def generate_embeddings(text):
    emBed = SentenceTransformer('all-MiniLM-L6-v2')
    return emBed.encode(text).tolist()  # Convert numpy array to list for CSV storage

# Adding in a new column 'embeddings' to the DataFrame, where each row contains the vector representation of the corresponding text in the 'lyrics' column.
file['embeddings'] = file['Lyrics'].apply(generate_embeddings)

file.to_csv(output_path, index=False)
print("Files Converted to Vectors and saved to finalDatasetSample_Embeddings_Vectored.csv")