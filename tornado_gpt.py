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

if len(sys.argv) < 2:
  print(f"Please provide a vibe in the format 'summer bbq party'")
  sys.exit(1)

SPOTIFY_CLIENT_ID = "841bb956c9984faa9b64705535a26429"
SPOTIFY_CLIENT_SECRET = "5eee3dc60d5a416887a4c5e4f0e2ff43"
OPENAPI_API_KEY = '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'

spotify_client = chatspot.login(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, OPENAPI_API_KEY)

vibe = (' '.join(sys.argv[1:]))
#
# To Cut and Paste to File 
#
import re
import json
songs = chatspot.songs_by_vibe(vibe, model="gpt-3.5-turbo-0301")
validated_songs = chatspot.lookup_songs(spotify_client, songs)

c = Counter()
validated_tracks_for_tim = []
for s in validated_songs:
  if len(validated_tracks_for_tim) == 5:
    break
  s_uri = s['uri']
  if s_uri == 'NOTFOUND':
    continue
  else:
    validated_track = {"title": f"{s['title']}", "artist": f"{s['artist']}", "isrc": f"{s['isrc']}", "uri": f"{s_uri}"}
    validated_tracks_for_tim.append(validated_track)
  if 'artist_genres' in s:
    for g in s['artist_genres']:
      c[g] += 1
try:
  genre = c.most_common(1)[0][0]
except:
  genre = None

#filename = re.sub("[^a-zA-Z0-9]","",vibe)
data = {"vibe_input": vibe, "valid_track_count": len(validated_tracks_for_tim), "seed_genre": genre, "seed_tracks": validated_tracks_for_tim}

if not validated_tracks_for_tim:
  sys.stderr.write("No valid songs found, try another vibe. Alternatively GPT may not be working.\n")
  sys.exit(1)
result=json.dumps(data, indent=2)
print(result)
