import whisper
from pathlib import Path
from typing import List
from .transcribed_segment import TranscribedSegment
import logging
import torch

# asdf
class WhisperProcessor:
    def __init__( self, 
            logger: logging.Logger,
            model_size: str = "base",
            threads: int = 1,
            device: str = "cpu"
            ):
        logger.info(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        self.thread = threads
        self.device = torch.device(device)
        self.LOG = logger
    
    def transcribe(self, audio_path: str) -> List[TranscribedSegment]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        self.LOG.info(f"Transcribing audio: {audio_path}")

        result = self.model.transcribe(
            str(audio_path),
            fp16=False if self.device == "cpu" else True,
            verbose=False,
            language="en",
            word_timestamps=True,
            condition_on_previous_text=True ,
            )
        
        segments = []
        last_end = 0.0
        for i, segment in enumerate(result["segments"]):
            segments.append(TranscribedSegment(
                start=segment["start"],
                end=segment["end"],
                speaker=None,
                text=segment["text"].strip(),
                confidence=segment.get("confidence", 0.0),
                language=result.get("language", ""),
                words=segment.get("words", []),
                segment_id=i,
                speaker_confidence=0.0
            ))
        
        return segments
