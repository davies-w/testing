#!/usr/bin/env python3

#!pip install -q 'git+https://github.com/davies-w/chatspot.git'
#!pip install numpy
#!pip install scipy
#!pip install pandas
#!pip install plotly
#!pip install scikit-learn

import np_json
import gpt

params = np_json.from_stdin()

seed_genre_and_tracks = gpt.gpt(params)

np_json.to_stdout(seed_genre_and_tracks)

