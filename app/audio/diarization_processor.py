import os
os.environ["SPEECHBRAIN_USE_SYMLINKS"] = "False"
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils.hook import ProgressHook
from typing import List, Optional
from pathlib import Path
import logging

class DiarizationProcessor:
    def __init__(self, 
                 hf_token: str,
                 device: str = "cpu",
                 num_speakers: Optional[int] = None,
                 logger: logging.Logger = None):
        self.LOG = logger
        self.device = torch.device(device)
        self.num_speakers = num_speakers
        
        self.LOG.info("Loading speaker embedding model...")
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )
        
        self.LOG.info("Loading diarization pipeline...")    
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    
    def process(self, audio_path: str, progress_hook: bool = True) -> Pipeline:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.LOG.info("Performing speaker diarization...")
        if progress_hook:
            with ProgressHook() as hook:
                return self.pipeline(
                    str(audio_path),
                    num_speakers=self.num_speakers,
                    hook=hook
                )
        return self.pipeline(
            str(audio_path),
            num_speakers=self.num_speakers
        )
