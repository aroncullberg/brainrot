import os
os.environ["SPEECHBRAIN_USE_SYMLINKS"] = "False"
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
from pyannote.audio.pipelines.utils.hook import ProgressHook

@dataclass
class TranscribedSegment:
    """A segment of transcribed audio with speaker diarization"""
    start: float              # Start time in seconds
    end: float               # End time in seconds
    speaker: str             # Speaker identifier
    text: str               # Transcribed text
    confidence: float       # Whisper's confidence score
    language: str           # Detected language
    words: List[Dict]       # Word-level timing if available
    segment_id: int         # Sequential ID for ordering
    speaker_confidence: float  # Diarization confidence

class AudioProcessor:
    """Process audio files for transcription and speaker diarization"""
    
    def __init__(self,
                 whisper_model_size: str = "base",
                 hf_token: str = None,
                 num_speakers: Optional[int] = None,
                 device: str = "cpu"):
        """
        Initialize the audio processor.
        
        Args:
            whisper_model_size: Size of Whisper model ("tiny", "base", "small", "medium", "large")
            hf_token: HuggingFace access token for pyannote.audio
            num_speakers: Optional number of speakers (if known)
            device: Computing device ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        
        # Initialize speaker embedding model
        print("Loading speaker embedding model...")
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )
        
        # Initialize Whisper model
        print(f"Loading Whisper model ({whisper_model_size})...")
        self.whisper_model = whisper.load_model(whisper_model_size)
        
        # Initialize diarization pipeline
        if not hf_token:
            raise ValueError("HuggingFace token is required for speaker diarization")
        
        print("Loading diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        self.num_speakers = num_speakers
        
    def process_file(self, audio_path: str, progress_hook: bool = True) -> List[TranscribedSegment]:
        """
        Process an audio file for transcription and speaker diarization.
        
        Args:
            audio_path: Path to audio file
            progress_hook: Whether to show progress information
            
        Returns:
            List of TranscribedSegment objects
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Transcribe with Whisper
        print("Transcribing audio...")
        whisper_result = self.whisper_model.transcribe(str(audio_path))
        
        # Perform diarization
        print("Performing speaker diarization...")
        if progress_hook:
            with ProgressHook() as hook:
                diarization = self.pipeline(
                    str(audio_path), 
                    num_speakers=self.num_speakers,
                    hook=hook
                )
        else:
            diarization = self.pipeline(
                str(audio_path), 
                num_speakers=self.num_speakers
            )
        
        print("Processing segments...")
        segments = []
        
        # Combine results
        for i, w_segment in enumerate(whisper_result["segments"]):
            start = w_segment["start"] 
            end = w_segment["end"]
            
            # Find matching speaker
            speaker = None
            speaker_conf = 0.0
            
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= start <= turn.end:
                    speaker = spk
                    speaker_conf = turn.confidence if hasattr(turn, 'confidence') else 0.0
                    break
                    
            segment = TranscribedSegment(
                start=start,
                end=end,
                speaker=speaker,
                text=w_segment["text"].strip(),
                confidence=w_segment.get("confidence", 0.0),
                language=whisper_result.get("language", ""),
                words=w_segment.get("words", []),
                segment_id=i,
                speaker_confidence=speaker_conf
            )
            
            segments.append(segment)
        
        print(f"Processed {len(segments)} segments")
        return segments

    def process_file_whisper_only(self, audio_path: str) -> List[TranscribedSegment]:
        """
        Process an audio file for transcription only (no speaker diarization).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of TranscribedSegment objects
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Transcribe with Whisper
        print("Transcribing audio...")
        whisper_result = self.whisper_model.transcribe(str(audio_path))
        
        segments = []
        
        # Process segments
        for i, w_segment in enumerate(whisper_result["segments"]):
            segment = TranscribedSegment(
                start=w_segment["start"],
                end=w_segment["end"],
                speaker=None,  # No speaker information
                text=w_segment["text"].strip(),
                confidence=w_segment.get("confidence", 0.0),
                language=whisper_result.get("language", ""),
                words=w_segment.get("words", []),
                segment_id=i,
                speaker_confidence=0.0
            )
            
            segments.append(segment)
        
        print(f"Processed {len(segments)} segments")
        return segments