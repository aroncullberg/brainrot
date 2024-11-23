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

        exit()
        
        return segments

    def merge_segments(self, segments, diarization_output):
        for segment in segments:
            for turn, _, speaker in diarization_output.itertracks(yield_label=True):
                # self.LOG.info(f"Turn: {turn.start} - {turn.end} | Segment: {segment.start} - {segment.end}")
                # self.LOG.info(f"Speaker: {speaker}")
                # self.LOG.info(f"Text: {segment.text}")
                # print("\n\n\n\n")
                if abs(turn.start - segment.start) < 0.1 or abs(turn.end - segment.end) < 0.1:
                # if turn.start <= segment.start <= turn.end - 0.2: 
                    segment.speaker = speaker
                    segment.speaker_confidence = (
                        turn.confidence if hasattr(turn, 'confidence') else 0.0
                    )

                    # self.LOG.info(f"MATCH")
                    # print("\n\n\n\n")
                    break
            # input("continue?")




    