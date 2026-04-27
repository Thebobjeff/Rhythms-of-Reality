import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

auth_manager = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Simple search to pull metadata (tags)
track_name = "Bohemian Rhapsody"
results = sp.search(q=track_name, limit=1, type='track')

if results['tracks']['items']:
    track = results['tracks']['items'][0]
    # Assuming 'track' is the object you got from your search results
    artist_id = track['artists'][0]['id']
    artist_info = sp.artist(artist_id)

    # Genres are returned as a list, e.g., ['rock', 'classic rock']
    genres = artist_info.get('genres', [])

    print(f"Artist: {artist_info['name']}")
    print(f"Genres: {', '.join(genres) if genres else 'No genres found'}")
    # These are the standard "tags" available in a track object
    tags = {
        "Name": track.get('name'),
        "Artist": track['artists'][0].get('name'),
        "Album": track['album'].get('name'),
        "Release Date": track['album'].get('release_date'),
        "Popularity": track.get('popularity'),
        "Explicit": track.get('explicit'),
        "Track Number": track.get('track_number'),
        "Disc Number": track.get('disc_number')
    }

    print("--- Song Tags ---")
    for key, value in tags.items():
        print(f"{key}: {value}")
else:
    print("No results found.")