from BeatNet.BeatNet import BeatNet
import pydub 
from pydub import silence
import librosa

def beatnet(params):
  whole = params.get("whole", None)
  drums = params.get("drums", None)
  drumoffset = params.get("drumoffset", None)
  regime = params.get("regime", 1)
  bars = params.get("bars", None) 

  if drumoffset:
      pydub.audio
      drumoffset = silence.detect_leading_silence(drums_audiosegment, silence_threshold=-45, chunk_size=10)

  print(drumoffset)
  
  estimator = BeatNet(regime, mode='offline', inference_model='DBN', plot=[], thread=False)

  bn_data = []
  if whole and not drums:
    bn_data = estimator.process(whole)

  if not whole and drums:
    bn_data = estimator.process(drums)
    
  if whole and drums:
    unused_bn_data = estimator.process(whole)
    bn_data = estimator.process(drums)

  print(bn_data)
  if bars:
    return [{'start': beat[0]} for beat in bn_data if beat[1] == 1.0]
  else:
    return [{'start': beat[0]} for beat in bn_data if beat[1] == 1.0]

 
