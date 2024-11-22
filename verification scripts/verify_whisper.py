import whisper
import os
from datetime import datetime
import time

def generate_captions(audio_path, model_name="base", output_formats=["srt"]):
    """
    Generate captions for an audio file
    
    Parameters:
    - audio_path: path to WAV file
    - model_name: whisper model size ("tiny", "base", "small", "medium", "large")
    - output_formats: list of formats to save ("srt", "vtt", "txt")
    """
    try:
        # Load the model
        print(f"Loading {model_name} model...")
        model = whisper.load_model(model_name)
        
        # Start timing
        start_time = time.time()
        
        # Transcribe
        print("Transcribing audio...")
        result = model.transcribe(audio_path, verbose=True)
        
        # Create output filename base (without extension)
        filename_base = os.path.splitext(audio_path)[0]
        
        # Save in requested formats
        for format in output_formats:
            output_path = f"{filename_base}_{model_name}.{format}"
            
            if format == "srt":
                # Save as SRT
                with open(output_path, "w", encoding="utf-8") as srt_file:
                    for i, segment in enumerate(result["segments"], start=1):
                        # Format timestamps
                        start = str(datetime.utcfromtimestamp(segment["start"]).strftime('%H:%M:%S,%f'))[:-3]
                        end = str(datetime.utcfromtimestamp(segment["end"]).strftime('%H:%M:%S,%f'))[:-3]
                        
                        # Write SRT segment
                        srt_file.write(f"{i}\n")
                        srt_file.write(f"{start} --> {end}\n")
                        srt_file.write(f"{segment['text'].strip()}\n\n")
            
            elif format == "txt":
                # Save as plain text
                with open(output_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(result["text"])
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Output files saved with prefix: {filename_base}_{model_name}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Example usage
    audio_file = "Computer Ethics.wav"  # Replace with your WAV file path
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found!")
    else:
        generate_captions(
            audio_path=audio_file,
            model_name="tiny",  # Use "tiny" for faster testing, "base" for better accuracy
            output_formats=["srt", "txt"]  # Generate both SRT and plain text
        )