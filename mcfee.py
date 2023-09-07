# mcfee.py
# 
# requires librosa, numpy, scipy
#
import librosa
import numpy as np
import scipy
import sklearn.cluster

def mcfee(y, sr, beats_in_seconds=None, dimensions=6, clusters=3):

  BINS_PER_OCTAVE = 12 * 3
  N_OCTAVES = 7

  C = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr,
                                     bins_per_octave=BINS_PER_OCTAVE,
                                     n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
                                     ref=np.max)

  # these will already be fixed if librosa, but doesn't matter if we redo.
  beat_times = [x['start'] for x in beats_in_seconds]
  beat_times = np.array(beat_times)
  # Get frames of beats, clipped to zero frame.
  beats = librosa.util.fix_frames(librosa.time_to_frames(beat_times, sr=sr), x_min=0)
  # Now get the fixed beat_times (basically 0.0 again)
  beat_times = librosa.frames_to_time(beats, sr=sr) #

  Csync = librosa.util.sync(C, beats, aggregate=np.median)

  R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',sym=True)

  df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
  Rf = df(R, size=(1, 7))

  mfcc = librosa.feature.mfcc(y=y, sr=sr)
  Msync = librosa.util.sync(mfcc, beats)

  path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
  sigma = np.median(path_distance)
  path_sim = np.exp(-path_distance / sigma)

  R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

  deg_path = np.sum(R_path, axis=1)
  deg_rec = np.sum(Rf, axis=1)

  mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

  A = mu * Rf + (1 - mu) * R_path

  L = scipy.sparse.csgraph.laplacian(A, normed=True)

  evals, evecs = scipy.linalg.eigh(L)
  evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
  Cnorm = np.cumsum(evecs**2, axis=1)**0.5

  k = dimensions
  X = evecs[:, :k] / Cnorm[:, k-1:k]
  num_clusters = clusters
  KM = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=100)

  seg_ids = KM.fit_predict(X)

  bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

  bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
  bound_segs = list(seg_ids[bound_beats])
  bound_frames = beats[bound_beats]

  # Make sure we cover to the end of the track
  bound_frames = librosa.util.fix_frames(bound_frames,
                                         x_min=None,
                                         x_max=C.shape[1]-1)

  bound_times = librosa.frames_to_time(bound_frames, sr=sr)
  freqs = librosa.cqt_frequencies(n_bins=C.shape[0],
                                  fmin=librosa.note_to_hz('C1'),
                                  bins_per_octave=BINS_PER_OCTAVE)

  # return these values, then we can plot together (and the spectragrams etc)
  return y, C, Csync,  X, sr, BINS_PER_OCTAVE, freqs, num_clusters, beat_times, bound_times, bound_segs
