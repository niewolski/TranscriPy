import os
from pathlib import Path
from typing import Optional, Union

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class Transcriber:
    # main class for transcribing audio and video files to text
    
    def __init__(self, model_size: str = "base"):
        # initialize transcriber with whisper model
        # args: model_size - size of whisper model to use (tiny, base, small, medium, large)
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "whisper is not installed. install it with: pip install openai-whisper"
            )
        
        self.model_size = model_size
        self.model = None
    
    def _load_model(self):
        # load whisper model if not already loaded
        if self.model is None:
            self.model = whisper.load_model(self.model_size)
    
    def _extract_audio_from_video(self, video_path: Union[str, Path]) -> str:
        # extract audio from video file
        # args: video_path - path to video file
        # returns: path to extracted audio file
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "moviepy is not installed. install it with: pip install moviepy"
            )
        
        video = VideoFileClip(str(video_path))
        audio_path = str(video_path).replace('.mp4', '_temp_audio.wav')
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return audio_path
    
    def _is_video_file(self, file_path: Union[str, Path]) -> bool:
        # check if file is a video file
        return str(file_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    def transcribe(
        self, 
        file_path: Union[str, Path], 
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        # transcribe audio or video file to text
        # args: file_path - path to audio (mp3) or video (mp4) file
        #       output_file - optional path to save transcription text file
        # returns: transcribed text as string
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")
        
        # load model if needed
        self._load_model()
        
        # handle video files - extract audio first
        audio_path = str(file_path)
        temp_audio = None
        
        if self._is_video_file(file_path):
            audio_path = self._extract_audio_from_video(file_path)
            temp_audio = audio_path
        
        try:
            # transcribe audio
            result = self.model.transcribe(audio_path)
            text = result["text"].strip()
            
            # save to file if output path provided
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            return text
        
        finally:
            # cleanup temporary audio file if created
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)

