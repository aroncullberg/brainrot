from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TranscribedSegment:
    start: float
    end: float
    speaker: str
    text: str
    confidence: float
    language: str
    words: List[Dict]
    segment_id: int
    speaker_confidence: float

