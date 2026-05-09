import pandas as pd
import os
import asyncio
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
BATCH_SIZE = 50
# --- 1. Configuration ---
# These are the target categories the LLM must choose from
GENRE_CATEGORIES = [
    "Pop", "Hip-Hop/Rap", "R&B/Soul", "Rock", "Country",
    "Dance/Electronic", "Latin", "Alternative", "Folk/Acoustic",
    "Gospel/Christian", "Metal/Hard Rock", "Reggae", "Jazz/Blues"
]

# --- 2. Initialize Models ---
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"))
openai_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"))
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")) #Reached my rate Limit for groq, its gonna sit this one out for now, but the structure is there to easily add it back in when I'm able to use it again

# Define the LangChain logic (LCEL)
prompt_template = ChatPromptTemplate.from_template("{request}")
parser = StrOutputParser()

# Create the chains
chains = {
    "Gemini": prompt_template | gemini_llm | parser,
    "OpenAI": prompt_template | openai_llm | parser,
    "Groq": prompt_template | groq_llm | parser
}

# --- 3. Core Logic Functions ---

async def run_chain(chain, request_text):
    """Invokes a single chain and handles potential errors."""
    try:
        response = await chain.ainvoke({"request": request_text})
        return response.strip()
    except Exception as e:
        print(f"    Error calling model: {e}")
        return "Unknown"

async def get_consensus_genre(artist_name, song_name):
    """Calls all chains for a specific Song/Artist pair and picks the best result."""
    
    request_content = f"""
    You are a music genre expert. Classify the song '{song_name}' by the artist '{artist_name}' 
    into EXACTLY ONE of these categories:
    {', '.join(GENRE_CATEGORIES)}
    
    Rules:
    1. Return ONLY the category name from the list provided.
    2. Do not include any explanation, intro text, or punctuation.
    3. If the artist spans multiple genres, choose the one that best fits this specific song.
    4. If completely unable to identify, return "Unknown".
    """
    
    # Run all three models simultaneously for this pair
    tasks = [run_chain(chain, request_content) for chain in chains.values()]
    results = await asyncio.gather(*tasks)
    
    # Filter results: Only keep those that exactly match our pre-defined list
    valid_results = [r for r in results if r in GENRE_CATEGORIES]
    
    if valid_results:
        # Return the most frequent valid result (majority vote)
        return max(set(valid_results), key=valid_results.count)
    
    return "Unknown"

async def run_batch_processing(pairs_list):
    """Iterates through unique Artist/Song pairs and fetches genres."""
    genre_mapping = {}
    # Split the list into chunks of BATCH_SIZE
    for i in range(0, len(pairs_list), BATCH_SIZE):
        batch = pairs_list[i : i + BATCH_SIZE]
        print(f"Processing batch {(i // BATCH_SIZE) + 1} (Items {i} to {i + len(batch)})...")
        
        # Create a list of tasks for the current batch
        # Each task is a consensus call (which itself runs 3 models)
        tasks = [get_consensus_genre(artist, song) for artist, song in batch]
        
        # Run the entire batch of 50 songs concurrently
        batch_results = await asyncio.gather(*tasks)
        
        # Map the results to the (Artist, Song) tuple
        for pair, result in zip(batch, batch_results):
            genre_mapping[tuple(pair)] = result
         
        await asyncio.sleep(1)  # Adjust as needed based on observed rate limits
        
    return genre_mapping

# --- 4. Main Execution Logic ---
# def process_dataframe(df, target_column='Genre'):
#     """
#     Identifies 'Unknown' values in the target column and fills them 
#     using Artist and Song info.
#     """
#     # 1. Identify rows where the target column is "Unknown"
#     mask = df[target_column] == "Unknown"
    
#     # 2. Get unique (Artist, Song) pairs to save API costs
#     unknown_pairs = df.loc[mask, ['Artist', 'Song']].drop_duplicates().values.tolist()
    
#     if not unknown_pairs:
#         print(f"No 'Unknown' values found in column '{target_column}'.")
#         return df

#     print(f"--- Fixing {len(unknown_pairs)} unique Artist-Song pairs ---")
    
#     # 3. Handle the Async Loop for different environments
#     try:
#         # Standard approach
#         loop = asyncio.get_event_loop()
#         new_genres_map = loop.run_until_complete(run_batch_processing(unknown_pairs))
#     except RuntimeError:
#         # Approach for Jupyter Notebooks / Interactive environments
#         import nest_asyncio
#         nest_asyncio.apply()
#         new_genres_map = asyncio.run(run_batch_processing(unknown_pairs))

#     # 4. Map the results back to the original dataframe
#     # We use axis=1 to look up the (Artist, Song) tuple in our results map
#     df.loc[mask, target_column] = df[mask].apply(
#         lambda row: new_genres_map.get((row['Artist'], row['Song']), "Unknown"), 
#         axis=1
#     )
    
#     return df
def process_dataframe(df, target_column='Genre'):
    """
    Identifies 'Unknown' values in the target column, fills them 
    using Artist and Song info, and removes duplicates.
    """
    # 1. Identify rows where the target column is "Unknown"
    mask = df[target_column] == "Unknown"
    
    # 2. Get unique (Artist, Song) pairs to save API costs
    unknown_pairs = df.loc[mask, ['Artist', 'Song']].drop_duplicates().values.tolist()
    
    if not unknown_pairs:
        print(f"No 'Unknown' values found in column '{target_column}'.")
        # Even if no unknowns, we should still ensure the existing DF is deduplicated
        return df.drop_duplicates(subset=['Artist', 'Song'])

    print(f"--- Fixing {len(unknown_pairs)} unique Artist-Song pairs ---")
    
    # 3. Handle the Async Loop
    try:
        loop = asyncio.get_event_loop()
        new_genres_map = loop.run_until_complete(run_batch_processing(unknown_pairs))
    except RuntimeError:
        import nest_asyncio
        nest_asyncio.apply()
        new_genres_map = asyncio.run(run_batch_processing(unknown_pairs))

    # 4. Map the results back to the original dataframe
    df.loc[mask, target_column] = df[mask].apply(
        lambda row: new_genres_map.get((row['Artist'], row['Song']), "Unknown"), 
        axis=1
    )

    # --- NEW: Deduplication Step ---
    # This removes any duplicate rows based on the combination of Artist and Song
    # 'keep=first' ensures we keep one instance and throw away the rest
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Artist', 'Song'], keep='first')
    final_count = len(df)
    
    print(f"Deduplication complete: Removed {initial_count - final_count} duplicate rows.")
    
    return df

# --- Usage ---

# Load your file
df = pd.read_csv("C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\hot100_Consolidated.csv")

# If you want to fix the 'Genre' column:
df = process_dataframe(df, target_column='Genre')

# If you want to fix the 'Tags' column:
# df = process_dataframe(df, target_column='Tags')

# Save the result
df.to_csv("C:\\Users\\devon\\Documents\\GitHub\\lang Chain\\Test Project- git\\data\\CSV\\hot100_Consolidated_Genres.csv", index=False)