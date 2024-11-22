# Create a test.py file:
import torch
import whisper
import moviepy.editor as mp

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())  # Should be False for CPU