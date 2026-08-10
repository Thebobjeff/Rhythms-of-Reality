This is a fantastic project. Combining semantic embeddings with sociological analysis of Hip-Hop is a very "pro" level use of AI.

The "Managed Democracy" part of your code is particularly clever—using a multi-LLM ensemble to verify genre labels is a high-level data science technique.

Here is a revamped `README.md` that makes the project "come to life" by blending the academic depth of your poster with the technical details of your code.

---

# 🎤 CISE: Semantic Drift in Hip-Hop Lyrics 🧬

**Analyzing the evolution of culture through the lens of High-Dimensional Vector Embeddings.**
 

## 🌟 Overview
Does Hip-Hop reflect the world, or does it change it? This project utilizes **Computational Linguistics** and **Machine Learning** to track the "Semantic Drift" of Hip-Hop lyrics from 1990 to 2024. 

By converting lyrics into high-dimensional vectors (embeddings), we can mathematically measure how lyrical themes have shifted in response to major cultural ruptures like the rise of Gangsta Rap, the Black Lives Matter movement, and the COVID-19 pandemic.

---

## 🚀 Key Features

### 🤖 1. "Managed Democracy" Genre Classification
To ensure data integrity, we don't trust just one AI. We use a **triple-model ensemble** (OpenAI, Gemini, and Groq) to label song genres. The system only confirms a genre if a majority consensus is reached, effectively eliminating single-model bias.

### ⚡ 2. High-Performance Vectorization
Utilizing the `SentenceTransformer` (`all-MiniLM-L6-v2`), we process thousands of songs in optimized batches of 50, transforming raw text into semantic coordinates.

### 📉 3. Semantic Drift Analysis
*   **Annual Centroids:** We calculate the "average meaning" of Hip-Hop for every year.
*   **Cosine Similarity:** We measure the distance between years to identify moments of "Notable Shift" (≥ 1 Standard Deviation from the mean).
*   **Diversity Tracking:** Measuring how the "spread" of lyrical themes has expanded or contracted over decades.

---

## 🛠️ Technical Pipeline

### **Stage 1: Data Consolidation & Voting**
We merge outputs from multiple LLMs to create a gold-standard dataset.
```python
# The "Managed Democracy" Logic
def calculate_majority(row):
    if row['Genre_OpenAI'] == row['Genre_Gemini']:
        return row['Genre_OpenAI']
    if row['Genre_OpenAI'] == row['Genre_Groq']:
        return row['Genre_OpenAI']
    if row['Genre_Gemini'] == row['Genre_Groq']:
        return row['Genre_Gemini']
    return "Unknown"
```

### **Stage 2: Semantic Embedding Generation**
Transforming lyrics into 384-dimensional space.
```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Lyrics'].tolist(), batch_size=50, show_progress_bar=True)
```

### **Stage 3: Genre Filtering**
Precision filtering to isolate the Hip-Hop/Rap core from the Billboard Hot 100.
```python
hiphop_df = df[df["Genre"] == "Hip-Hop/Rap"].reset_index(drop=True)
```

---

## 📊 Findings (from the CISE Study)
*   **The Stability of the 2000s:** Hip-Hop was lyrically most stable during the mid-2000s "Crunk" era, reflecting a period of commercial uniformity.
*   **The Great Diversification:** Since the 2010s, the "semantic spread" has widened significantly—meaning Hip-Hop lyrics today are more diverse and varied than at any other point in history.
*   **Cultural Mirrors:** Major dips in semantic similarity align perfectly with the deaths of 2Pac/Biggie (1996-1998) and the global pandemic (2020).

---

## 📥 Installation

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/yourusername/semantic-drift-hiphop.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install pandas sentence-transformers pathlib openai
   ```
3. **Environment Setup:**
   Create a `.env` file and add your API keys:
   ```env
   GROQ_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```

---

## 👨‍💻 Contributors
*   **Lead Researchers:** Leyanna Daniels, Devonte Edward
*   **Mentor:** Thitima Srivatanakul
*   **Institution:** CSTEP & STEP / York College (CUNY)

---
*Developed as part of the SEMANTIC DRIFT IN HIP-HOP LYRICS study.*
