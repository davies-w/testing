import mcfee
import np_json
import librosa

params = np_json.from_stdin()

#
# read all the files as mono wavs (default of Librosa load)
#
y, sr = librosa.load(params["harmony_stem_files"].pop(0))
for file in params["harmony_stem_files"]:
  y1, _ = librosa.load(file)
  y += y1

#
# Use a provide beat array of dicts, else use Librosa
#
if "beats" in params:
  beats = params["beats"]
else:
  _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
  beat_times = librosa.util.fix_frames(beat_frames, x_min=0)
  beats = [{'start': t} for t in librosa.frames_to_time(beat_times, sr=sr)]

dims =  params.get("dimensions", 6) 
k = params.get("clusters", 3)

_, _, _, _, _, _, _, _, _, times, segs = mcfee.mcfee(y, sr, beats, dims, k)

sections = [{"start": round(t, 3), "label": l }  for t, l in zip(times, segs)]

np_json.to_stdout(sections)
