#!/usr/bin/env python3

#!pip install -q 'git+https://github.com/davies-w/chatspot.git'
#!pip install numpy
#!pip install scipy
#!pip install pandas
#!pip install plotly
#!pip install sklearn
#!pip install scikit-learn

import chatspot
import sys
from collections import Counter
import np_json


params = np_json.from_stdin()

SPOTIFY_CLIENT_ID = "841bb956c9984faa9b64705535a26429"
SPOTIFY_CLIENT_SECRET = "5eee3dc60d5a416887a4c5e4f0e2ff43"
OPENAPI_API_KEY = "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"
MODEL = "gpt-3.5-turbo"
spotify_client = chatspot.login(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, OPENAPI_API_KEY)


#
# To Cut and Paste to File 
#
import re
import json
songs = chatspot.songs_by_vibe(params["vibe"], model=MODEL)
validated_songs = chatspot.lookup_songs(spotify_client, songs)

validated_songs = [s  for s in validated_songs if s['uri'] != 'NOTFOUND']

genre = None
genres = Counter()
for s in validated_songs:
  if 'artist_genres' in s:
    for g in s['artist_genres']:
      genres[g] += 1

if len(genres) > 0:
  genre = genres.most_common()[0][0]


validated_tracks = []
for s in validated_songs:
  validated_tracks += [ {"title": s['title'], 
                        "artist": s['artist'], 
                        "isrc": s['isrc'], 
                         "uri": s['uri']}]

validated_tracks = validated_tracks[0:4 if genre else 5]
seed_genre_and_tracks  = {"valid_track_count": len(validated_tracks),
                         "seed_genre": genre, 
                         "seed_tracks": validated_tracks}

np_json.to_stdout(seed_genre_and_tracks)

