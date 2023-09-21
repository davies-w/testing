from BeatNet.BeatNet import BeatNet
from pydub import AudioSegment
from pydub import silence
import librosa
import numpy as np

def audiosegment_to_np_norm(audiosegment):
  """
  Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
  where each value is in range [-1.0, 1.0].
  Returns tuple (audio_np_array, sample_rate).
  """
  return np.array(audiosegment.get_array_of_samples(), dtype=np.float32).reshape((-1, audiosegment.channels)) / (
                  1 << (8 * audiosegment.sample_width - 1)), audiosegment.frame_rate


def audiosegment_to_wav(audiosegment):
  return audiosegment_to_np_norm(audiosegment)


def beatnet(params):
  print(params)

  full = params.get("full", None)
  drums = params.get("drums", None)
  drumoffset = params.get("drumoffset", None)
  beatnet_selector = params.get("beatnet_selector ", 1)
  bars = params.get("bars", None) 
 
  if beatnet_selector == "librosa_drums" or  beatnet_selector == "librosa_full":
    if beatnet_selector == "librosa_full":
      y, sr = librosa.load(full, offset=0.0)
    if beatnet_selector == "librosa_drums":
      y, sr = librosa.load(drums, offset=0.0)
      
    unused_tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beats = librosa.util.fix_frames(beats, x_min=0)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return [{'start': x} for x in beat_times]
    
    
  estimator = BeatNet(beatnet_selector, mode='offline', inference_model='DBN', plot=[], thread=False)
    
  drumoffset_msecs = 0
  if drumoffset and drums:
    drums_audiosegment = AudioSegment.from_file(drums)
    drumoffset_msecs = silence.detect_leading_silence(drums_audiosegment, silence_threshold=-45, chunk_size=10)

  if drums:
    drums_audiosegment = AudioSegment.from_file(drums)
    trimmed_drums_audiosegment = drums_audiosegment[drumoffset_msecs:]
    trimmed_drums_mono_audiosegment = trimmed_drums_audiosegment.split_to_mono()
    trimmed_drums_mono_audiosegment = trimmed_drums_mono_audiosegment[0]
    trimmed_drums_mono_audiosegment = trimmed_drums_mono_audiosegment.set_frame_rate(22050)
    trimmed_drums_mono_wav, _ = audiosegment_to_wav(trimmed_drums_mono_audiosegment)
    trimmed_drums_mono_wav = trimmed_drums_mono_wav.astype('float32')
    
  if full:
    full_audiosegment = AudioSegment.from_file(full)
    trimmed_full_audiosegment = full_audiosegment[drumoffset_msecs:]
    trimmed_full_mono_audiosegment = trimmed_full_audiosegment.split_to_mono()
    trimmed_full_mono_audiosegment = trimmed_full_mono_audiosegment[0]
    trimmed_full_mono_audiosegment = trimmed_full_mono_audiosegment.set_frame_rate(22050)
    trimmed_full_mono_wav, _ = audiosegment_to_wav(trimmed_full_mono_audiosegment)
    trimmed_full_mono_wav = trimmed_full_mono_wav.astype('float32')

  if full and not drums:
    bn_data = estimator.process(trimmed_full_mono_wav)
  if  drums and not full:
    bn_data = estimator.process(trimmed_drums_mono_wav)
  if  full and drums: 
    unused_bn_data = estimator.process(trimmed_full_mono_wav)
    bn_data = estimator.process(trimmed_drums_mono_wav)

  if not bars:
    return [{'start': x[0]+(drumoffset_msec/1000)} for x in bn_data ]
  else:
    return [{'start': x[0]+(drumoffset_msec/1000)} for x in bn_data if x[1] == 1.0]
