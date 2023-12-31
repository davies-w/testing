import chatspot
import sys
import os
from collections import Counter
import re
import json


def gpt(params):
  SPOTIFY_CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
  SPOTIFY_CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
  OPENAPI_API_KEY = os.environ["OPENAI_API_KEY"]
  OPENAI_MODEL = os.environ["OPENAI_MODEL"]
  spotify_client = chatspot.login(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, OPENAPI_API_KEY)


  #
  # To Cut and Paste to File 
  #

  songs = chatspot.songs_by_vibe(params["vibe"], model=OPENAI_MODEL)
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
  return  {"valid_track_count": len(validated_tracks),
           "seed_genre": genre, 
           "seed_tracks": validated_tracks}
  
