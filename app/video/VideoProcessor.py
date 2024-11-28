from enum import Enum
import gc
import cv2
import numpy as np
from typing import Generator, Optional, List, Tuple

from audio.transcribed_segment import TranscribedSegment

class VideoProcessor:
    def __init__(self, video_path: str, batch_size: int = 32, segments: List[TranscribedSegment] = None):
        """
        Initialize the video processor.
        
        Args:
            video_path (str): Path to the video file
            batch_size (int): Number of frames to process at once
            segments (List[Tuple[str, float, float]]): List of (text, start_time, end_time) tuples
        """
        self.video_path = video_path
        self.batch_size = batch_size
        self.segments = segments or []
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()
            
    @property
    def video_info(self) -> dict:
        """Get basic information about the video."""
        return {
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

    def last_segment_end(self) -> float:
        """Get the end time of the last segment."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end
    
    def get_speaker_for_timestamp(self, timestamp: float) -> str:
        """
        Get the color that should be displayed at a given timestamp.
        
        Args:
            timestamp (float): Current video timestamp in seconds
            
        Returns:
            Optional[Tuple[int, int, int]]: Color to display, or None if no color should be shown
        """
        for segment in self.segments:
            if segment.start <= timestamp <= segment.end:
                return segment.speaker
        return None

    def get_text_for_timestamp(self, timestamp: float) -> str:
        """
        Get the text that should be displayed at a given timestamp.
        
        Args:
            timestamp (float): Current video timestamp in seconds
            
        Returns:
            str: Text to display, or empty string if no text should be shown
        """
        # for text, start, end in self.segments:
        for segment in self.segments:
            if segment.start <= timestamp <= segment.end:
                return segment.text
        return ""
            
    def frame_generator(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generate frames one at a time with their timestamps.
        
        Yields:
            Tuple[np.ndarray, float]: (frame, timestamp) pairs
        """
        frame_count = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # reset video to benigin
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                print("Video has ended, looping back to beginning")
                
            # Calculate current timestamp in seconds
            timestamp = frame_count / self.video_info['fps']

            if timestamp > self.last_segment_end():
                break
            
            yield frame, timestamp
            frame_count += 1
            
    def batch_generator(self, progress_callback = None) -> Generator[List[Tuple[np.ndarray, float]], None, None]:
        """
        Generate batches of frames with their timestamps.
        
        Yields:
            List[Tuple[np.ndarray, float]]: Batch of (frame, timestamp) pairs
        """
        batch = []
        frames_processed = 0
        total_frames = self.video_info['frame_count']
        for frame_data in self.frame_generator():
            batch.append(frame_data)
            frames_processed += 1

            if progress_callback:
                progress_callback(frames_processed, total_frames)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                gc.collect()
                
        if batch:
            yield batch


class Speaker(Enum):
    SPEAKER_00 = "SPEAKER_00"
    SPEAKER_01 = "SPEAKER_01"

class Color(Enum):
    '''
    Enum for color values.
    is in BGR format 
    '''
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    ORANGE = (0, 165, 255)
    BLACK = (0, 0, 0)


def wrap_text(text: str, font, font_scale: float, thickness: int, max_width: int) -> list:
    """
    Wrap text into multiple lines based on max width.
    
    Args:
        text (str): Text to wrap
        font: CV2 font
        font_scale (float): Font scale
        thickness (int): Text thickness
        max_width (int): Maximum width in pixels
        
    Returns:
        list: List of wrapped text lines
    """
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        word_size = cv2.getTextSize(word + ' ', font, font_scale, thickness)[0]
        if current_width + word_size[0] <= max_width:
            current_line.append(word)
            current_width += word_size[0]
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_size[0]
            
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def process_frame(frame: np.ndarray, timestamp: float, text: str, speaker: str, target_size: tuple = (720, 1280)) -> Optional[np.ndarray]:
    """
    Process a frame with text overlay.
    
    Args:
        frame (np.ndarray): Input frame
        timestamp (float): Current timestamp in seconds
        text (str): Text to overlay on the frame
        target_size (tuple): Desired output size (width, height)
        
    Returns:
        np.ndarray: Processed frame
    """
    # match speaker to color
    colors = {
        "SPEAKER_00": Color.ORANGE.value,
        "SPEAKER_01": Color.WHITE.value
    }
    color = colors.get(speaker, Color.GREEN.value) 
    # print(f'\n{speaker} | {color}\n\n')
    try:
        ### CROP ###
        current_ratio = frame.shape[0] / frame.shape[1]
        target_ratio = target_size[1] / target_size[0]

        if current_ratio > target_ratio:
            crop_height = int(frame.shape[1] * target_ratio)
            start = (frame.shape[0] - crop_height) // 2
            frame = frame[start:start + crop_height, :]
        else:
            crop_width = int(frame.shape[0] / target_ratio)
            start = (frame.shape[1] - crop_width) // 2
            frame = frame[:, start:start + crop_width]
        ############


        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.5
            thickness = 5
            borderthickness = 16
            height_padding = 16
            maxwidth = target_size[0] - 20  # 20 pixels from right edge

            lines = wrap_text(text, font, font_scale, thickness, maxwidth)

            line_height = cv2.getTextSize('A', font, font_scale, thickness)[0][1] + height_padding
            total_height = len(lines) * line_height

            line_widths = [cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]
            max_line_width = max(line_widths)

            text_block_x = (target_size[0] - max_line_width) // 2
            text_block_y = (target_size[1] - total_height) // 2
            
            current_y = text_block_y + line_height
            for line, line_width in zip(lines, line_widths):
                line_x = (target_size[0] - line_width) // 2
                # BORDER( ?)
                cv2.putText(resized, line, (line_x, current_y), font, font_scale, 
                           (0, 0, 0), borderthickness, cv2.LINE_AA) 
                
                # TExt 
                cv2.putText(resized, line, (line_x, current_y), font, font_scale, 
                           color=color, thickness=thickness, lineType=cv2.LINE_AA) 

                current_y += line_height

        return resized
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def post_process_video(input_video: str, audio_file: str, output_file: str, target_fps: int = 30, target_bitrate: str = "1M"):
    """
    Post-process video using FFmpeg to combine with audio, adjust FPS and bitrate.
    
    Args:
        input_video (str): Path to input video
        audio_file (str): Path to WAV file
        output_file (str): Path to final output file
        target_fps (int): Desired frame rate
        target_bitrate (str): Target bitrate (e.g., "1M" for 1 Mbps)
    """
    import subprocess
    
    # FFmpeg command to:
    # 1. Set output FPS
    # 2. Set video bitrate
    # 3. Add audio
    # 4. Ensure audio and video sync
    cmd = [
        'ffmpeg',
        '-i', input_video,           # Input video
        '-i', audio_file,            # Input audio
        '-c:v', 'libx264',           # Video codec
        '-preset', 'medium',         # Encoding speed preset
        '-b:v', target_bitrate,      # Video bitrate
        '-r', str(target_fps),       # Output frame rate
        '-c:a', 'aac',               # Audio codec
        '-b:a', '192k',              # Audio bitrate
        '-movflags', '+faststart',   # Enable fast start for web playback
        '-y',                        # Overwrite output file if exists
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully created final video: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg processing: {e}")
        raise

