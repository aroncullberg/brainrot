import os
os.environ["SPEECHBRAIN_USE_SYMLINKS"] = "False"
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from pyannote.audio.pipelines.utils.hook import ProgressHook

# Initialize models
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu")
)

whisper_model = whisper.load_model("base")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="Yhf_HwlCKZOwYgvDKVvinWbrTiXbmOqEgZAEBV"
)

# Process audio
audio_file = "audio.wav"

# 1. Transcribe with Whisper
whisper_result = whisper_model.transcribe(audio_file)
segments = whisper_result["segments"]

# 2. Perform diarization
with ProgressHook() as hook:
    diarization = pipeline(audio_file, hook=hook, num_speakers=2)

# 3. Combine results
for segment in segments:
    # Find matching speaker turn
    start = segment["start"]
    end = segment["end"]
    
    # Find overlapping diarization segment
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.start <= start <= turn.end:
            print(f"[{speaker}] {start:.1f}s - {end:.1f}s: {segment['text']}")
            break


"""
#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
"""