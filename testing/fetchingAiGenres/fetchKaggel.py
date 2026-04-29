import pandas as pd

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SET THE FILE PATH HERE
# This tells kagglehub which specific file inside the 'ludmin/billboard' dataset to load
file_path = "hot100.csv" 

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "ludmin/billboard",
  file_path,
)
 
# 2. Data Cleaning (Per your provided logic)
# Convert '-' to 0 and change types to integers
df['Weeks in Charts'] = df['Weeks in Charts'].replace('-', 0).astype(int)
df['Last Week'] = df['Last Week'].replace('-', 0).astype(int)

# Convert Date to datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# 3. Filter and Extract Year
df['Year'] = df['Date'].dt.year
df_filtered = df[df['Year'] >= 2025].copy()

# 4. Aggregate to find the best performance of each song per year
# We group by Year, Artist, and Song to find the highest rank (minimum value)
yearly_stats = df_filtered.groupby(['Year', 'Artist', 'Song']).agg({
    'Rank': 'min',
    'Date': 'min'
}).reset_index()

# 5. Rename for the final CSV format
yearly_stats.rename(columns={
    'Rank': 'Peak_Pos_That_Year', 
    'Date': 'Full_Release_Year'
}, inplace=True)

# 6. Sort by Year and Rank, then take the Top 100 for each year
top_100_final = yearly_stats.sort_values(['Year', 'Peak_Pos_That_Year']).groupby('Year').head(100)

# 7. Export to CSV
top_100_final.to_csv('hot100.csv', index=False)

print("File successfully created: hot100.csv")