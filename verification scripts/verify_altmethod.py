num_speakers = 2 #@param {type:"integer"}

language = 'English' #@param ['any', 'English']

model_size = 'tiny' #@param ['tiny', 'base', 'small', 'medium', 'large']

model_name = model_size
if language == 'English' and model_size != 'large':
  model_name += '.en'


from pathlib import Path
import whisper
import datetime

import subprocess

import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu"))

from pyannote.audio import Audio
from pyannote.core import Segment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np


audio_path = Path("fuck.wav")

model = whisper.load_model(model_size)

result = model.transcribe(str(audio_path), verbose=True)

segments = result["segments"]
print(segments)

with contextlib.closing(wave.open(str(audio_path),'r')) as f:
  frames = f.getnframes()
  rate = f.getframerate()
  duration = frames / float(rate)

print(frames)
print(rate)
print(duration)

# exit()

audio = Audio()

def segment_embedding(segment):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(str(audio_path), clip)
  return embedding_model(waveform[None])

print(segment_embedding(segments[0]))