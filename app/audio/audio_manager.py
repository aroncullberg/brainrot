import logging
from typing import List, Optional
from .whisper_processor import WhisperProcessor
from .diarization_processor import DiarizationProcessor
from .transcribed_segment import TranscribedSegment

class AudioManager:
    def __init__(self,
                 whisper_model_size: str = "base",
                 hf_token: str = None,
                 num_speakers: Optional[int] = None,
                 device: str = "cpu",
                 threads: int = 1,
                 logger: logging.Logger = None
                 ):

        self.whisper = WhisperProcessor(
            model_size=whisper_model_size,
            device=device,
            threads=threads,
            logger=logger)

        if hf_token:
            self.diarization = DiarizationProcessor(
                hf_token=hf_token,
                device=device,
                num_speakers=num_speakers,
                logger=logger
            )
        else:
            logger.warning("No HF token provided. Diarization will be disabled.")
            self.diarization = None
        self.LOG = logger
    
    def process_file(self, audio_path: str, progress_hook: bool = True) -> List[TranscribedSegment]:
        # Get transcription
        segments = self.whisper.transcribe(audio_path)
        for seg in segments:
            print(f'[ {seg.start} --> {seg.end} ] {seg.text}')
        
        if not self.diarization:
            return segments
            
        # Add speaker information
        diarization = self.diarization.process(audio_path, progress_hook)
        print(diarization)

        self.merge_segments(segments, diarization)
        # print([seg for seg in segments])

        return segments

    def merge_segments(self, segments, diarization_output):
        for segment in segments:
            max_overlap = 0
            best_speaker = None
            best_speaker_confidence = 0.0

            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                overlap_start = max(segment.start, turn.start)
                overlap_end = min(segment.end, turn.end)
                overlap = max(0, overlap_end - overlap_start)
                
                # Calculate overlap ratio relative to segment duration
                segment_duration = segment.end - segment.start
                overlap_ratio = overlap / segment_duration if segment_duration > 0 else 0
                
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    best_speaker = speaker
                    best_confidence = turn.confidence if hasattr(turn, 'confidence') else 0.0
            
            if max_overlap > seconds(0.2):
                segment.speaker = best_speaker
                segment.speaker_confidence = best_confidence


def seconds(second: float) -> float:
    '''
    Does nothing but to increase readability of code
    without using a comment
    '''
    return second

    