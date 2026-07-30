import torch
import whisper
from moviepy import VideoFileClip
from pathlib import Path
import config

class AudioProcessor:
    def __init__(self, model_name: str = config.WHISPER_MODEL_NAME):
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
        
        result = self.model.transcribe(audio_path, fp16=self.fp16)
        
        Path(audio_path).unlink(missing_ok=True)
        
        return {
            "text": result["text"],
            "segments": result["segments"]
        }

    def calculate_pacing(self, segments: list) -> dict:
        """Calculates speaking cadence (WPM) from Whisper segment timestamps."""
        if not segments:
            return {"wpm": 0, "total_words": 0, "status": "No Speech Detected"}

        total_words = sum(len(seg.get("text", "").split()) for seg in segments)
        start_time = segments[0].get("start", 0)
        end_time = segments[-1].get("end", 1)
        duration_minutes = max((end_time - start_time) / 60.0, 0.1)

        wpm = round(total_words / duration_minutes)

        if wpm > 160:
            status = "Fast Pace (Risk of low comprehension)"
        elif wpm < 110:
            status = "Slow Pace (Risk of low engagement)"
        else:
            status = "Optimal Cadence"

        return {
            "wpm": wpm,
            "total_words": total_words,
            "duration_minutes": round(duration_minutes, 1),
            "status": status
        }