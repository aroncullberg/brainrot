try:
    import os
    from collections import defaultdict
    import gc
    from pathlib import Path
    from typing import Optional, Tuple
    import time
    import cv2
    import numpy as np
    from tqdm import tqdm
    from video.VideoProcessor import VideoProcessor, process_frame, post_process_video
    from audio.transcribed_segment import TranscribedSegment
    from config.env import loadAPIKeys
    from config.argument_parser import ArgumentParser, BrainrotConfig
    from audio.audio_manager import AudioManager
    import logging
    from enum import Enum
except ImportError as e:
    print(f"Failed to import required module: {e}")
    raise

TEMP_VIDEO = "temp.mp4"

def main(config: BrainrotConfig):
    # Initialize logger
    logging.basicConfig(level=config.loglevel)
    LOG = logging.getLogger(__name__)

    # Load environment variables (API keys)
    loadAPIKeys(LOG)

    # Initialize processor
    audio = AudioManager(
        whisper_model_size=config.model,
        hf_token=os.environ.get("HF_TOKEN"),
        num_speakers=config.speakers,
        device=config.device,
        threads=config.threads,
        logger=LOG
    )

    segments = audio.process_file(config.audio)

    # Log results
    LOG.info("Speaker \t| Text")
    for seg in segments:
        LOG.info(f"{seg.speaker} \t| {seg.text}")

    BATCH_SIZE = 64
    with VideoProcessor(config.video, BATCH_SIZE, segments) as processor:
        info = processor.video_info
        target_size = (720, 1280)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            TEMP_VIDEO, 
            fourcc, 
            info['fps'],
            target_size,
            isColor=True  # Set to True since we're using colored text
        )

        total_frames = info['frame_count']
        fps = info['fps']
        processed_frames = 0

        # Create progress bar
        with tqdm(total=segments[-1].end * fps, unit='frames', 
                 desc='Processing video', 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames '
                           '[{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
            
            # Process video in batches
            start_time = time.time()
            for batch in processor.batch_generator():
                # Process each frame in the batch
                for frame, timestamp in batch:
                    # Get text for current timestamp
                    current_text = processor.get_text_for_timestamp(timestamp)
                    speaker = processor.get_speaker_for_timestamp(timestamp)
                    
                    # Process frame with text overlay
                    processed_frame = process_frame(frame, timestamp, current_text, speaker, target_size)
                    
                    if processed_frame is not None:
                        out.write(processed_frame)
                    
                    # Update progress bar
                    processed_frames += 1
                    pbar.update(1)
                    
                    # Update processing speed in postfix
                    elapsed_time = time.time() - start_time
                    fps = processed_frames / elapsed_time
                    pbar.set_postfix({'FPS': f'{fps:.2f}'})
                
                # Force memory cleanup after each batch
                gc.collect()
        out.release()

    try:
        print("\nPost-processing video with FFmpeg...")
        post_process_video(
            input_video = TEMP_VIDEO,
            audio_file = config.audio,
            output_file = config.output,
            target_fps = 60,
            target_bitrate = "6M"
        )
        os.remove(TEMP_VIDEO)
        print("Temporary files cleaned up")
    
    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    config = parser.parse()
    main(config)