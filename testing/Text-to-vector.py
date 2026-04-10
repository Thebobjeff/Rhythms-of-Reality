# First, install the library if you haven't: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Converting text to vectors using the Sentence Transformers (ST) library

sentences = ["Machine learning is fascinating!", "My fav color is blue."]

sentence1 = sentences[0]
sentence2 = sentences[1]

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to vectors
embeddings = model.encode(sentences)

embeddings1 = model.encode(sentence1)
embeddings2 = model.encode(sentence2)


print("Sentence Embeddings:")
# for sentence, embedding in zip(sentences, embeddings):
#     print(f"Sentence: {sentence}")
#     print(f"Vector (first 5 values): {embedding[:5]}")
#     print(f"Vector shape: {embedding.shape}")

 
print(f"Sentence: {sentence1}")
print(f"Vector (first 5 values): {embeddings1[:5]}")
print(f"Vector shape: {embeddings1.shape}")

print(f"Sentence: {sentence2}")
print(f"Vector (first 5 values): {embeddings2[:5]}")
print(f"Vector shape: {embeddings2.shape}")

# Calculate cosine similarity between the two sentences
similarity = cosine_similarity([embeddings1], [embeddings2])
print(f"Cosine Similarity: {similarity[0][0]}")