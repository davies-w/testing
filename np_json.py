import json
from json import JSONEncoder
import numpy as np
import sys

# NumPy JSON array encoder class
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def to_stdout(obj):
  json.dump(obj, sys.stdout, cls=NumpyArrayEncoder, indent=4)

def from_stdin():
  return json.load(sys.stdin)
