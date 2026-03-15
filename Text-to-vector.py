# First, install the library if you haven't: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

import numpy as np

# Converting text to vectors using the Sentence Transformers (ST) library

sentences = ["Machine learning is fascinating!", "AI is intriguing!"]

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to vectors
embeddings = model.encode(sentences)

print("Sentence Embeddings:")
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Vector (first 5 values): {embedding[:5]}")
    print(f"Vector shape: {embedding.shape}")
