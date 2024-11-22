import os
os.environ["SPEECHBRAIN_STRATEGY"] = "copy"
os.environ["SPEECHBRAIN_USE_SYMLINKS"] = "False"



import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils.hook import ProgressHook

device = torch.device("cpu")
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=device
)

# Your audio processing code here
audio_file = "path/to/audio.wav"  # Replace with your audio path
whisper_model = whisper.load_model("base")
