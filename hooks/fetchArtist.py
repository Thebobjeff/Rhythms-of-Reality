import billboard
import pandas as pd
import time

# 1. Configuration
chat_info = [
    ['Hip-Hop / R&B', 'hot-r-b-hip-hop-songs'],
    ['Rap', 'hot-rap-songs'],
    ['Rock & Alternative', 'hot-rock-songs'],
    ['Country', 'hot-country-songs'],
    ['Dance / Electronic', 'hot-dance-electronic-songs'],
    ['Latin', 'hot-latin-songs'],
    ['Pop', 'pop-songs'],
    ['Afrobeats', 'afrobeats-songs'],
]

years = range(1990, 1995) # 1990 to 2026
master_data = []

# 2. Execution Loop
for year in years:
    # We use the last Saturday of the year to simulate the "Year-End" state
    target_date = f"{year}-12-25" 
    print(f"--- Processing Year: {year} ---")

    for genre_title, slug in chat_info:
        try:
            # Fetch chart
            chart = billboard.ChartData(slug, date=target_date)
            print(f"  > Fetched {genre_title}")

            # Pull Top 100 entries and all fields from your image
            for entry in chart[:100]:
                master_data.append({
                    'Year': year,
                    'Genre_Category': genre_title,
                    'Chart_Name': chart.title,
                    'Artist': entry.artist,
                    'Song_Title': entry.title,
                    'Rank': entry.rank,
                    'Peak_Pos': entry.peakPos,
                    'Last_Pos': entry.lastPos,
                    'Weeks_on_Chart': entry.weeks,
                    'Is_New': entry.isNew,
                    'Image_URL': entry.image
                })
            
            # Short pause to be respectful to Billboard's servers
            time.sleep(0.5) 

        except Exception as e:
            # Some charts (like Afrobeats) didn't exist in 1990, so we skip errors
            print(f"  ! Skipping {genre_title} for {year}: (Chart may not exist yet)")

# 3. Create DataFrame and Export
df_final = pd.DataFrame(master_data)

# Save to the specific filename you requested
df_final.to_csv('billyartitst.csv', index=False)

print("\n" + "="*30)
print("File successfully created: billyartitst.csv")
print(f"Total rows captured: {len(df_final)}")