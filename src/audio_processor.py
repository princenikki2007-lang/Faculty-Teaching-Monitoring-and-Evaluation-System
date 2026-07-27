import torch
import whisper
from moviepy import VideoFileClip
from pathlib import Path
import config

class AudioProcessor:
    def __init__(self, model_name: str = config.WHISPER_MODEL_NAME):
        # Explicitly configure GPU/CUDA acceleration
        if torch.cuda.is_available():
            self.device = "cuda"
            self.fp16 = True
        else:
            self.device = "cpu"
            self.fp16 = False
            
        print(f"[AudioProcessor] Loading Whisper '{model_name}' on device: {self.device}")
        self.model = whisper.load_model(model_name, device=self.device)

    def extract_audio(self, video_path: str) -> str:
        """Extracts WAV audio from a video file."""
        video_path = Path(video_path)
        audio_path = config.UPLOAD_DIR / f"{video_path.stem}.wav"
        
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(audio_path), logger=None)
        clip.close()
        
        return str(audio_path)

    def transcribe(self, video_path: str) -> dict:
        """Extracts audio and transcribes it using GPU acceleration."""
        audio_path = self.extract_audio(video_path)
        print(f"[AudioProcessor] Transcribing on {self.device.upper()}...")
        
        # Transcribe audio using FP16 mode on CUDA for maximum performance
        result = self.model.transcribe(audio_path, fp16=self.fp16)
        
        # Cleanup temporary audio file
        Path(audio_path).unlink(missing_ok=True)
        
        return {
            "text": result["text"],
            "segments": result["segments"]
        }