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

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


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
        
        video_path = Path(video_path).resolve()
        
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")
        
        # create temp audio file in same directory as video
        audio_path = video_path.parent / f"{video_path.stem}_temp_audio.wav"
        audio_path = audio_path.resolve()
        
        # ensure parent directory exists
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
        video.close()
        
        # verify audio file was created
        if not audio_path.exists():
            raise FileNotFoundError(f"failed to extract audio to: {audio_path}")
        
        return str(audio_path.absolute())
    
    def _is_video_file(self, file_path: Union[str, Path]) -> bool:
        # check if file is a video file
        return str(file_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    def _convert_audio_to_wav(self, audio_path: Union[str, Path]) -> str:
        # convert audio file to wav format for better whisper compatibility
        # args: audio_path - path to audio file
        # returns: path to converted wav file
        audio_path = Path(audio_path).resolve()
        
        # if already wav, return as is
        if audio_path.suffix.lower() == '.wav':
            return str(audio_path.absolute())
        
        # try librosa first (doesn't require ffmpeg for many formats)
        if LIBROSA_AVAILABLE:
            try:
                wav_path = audio_path.parent / f"{audio_path.stem}_converted.wav"
                wav_path = wav_path.resolve()
                
                # load audio with librosa (handles mp3, wav, etc without ffmpeg)
                y, sr = librosa.load(str(audio_path), sr=16000)
                # save as wav
                sf.write(str(wav_path), y, sr)
                
                if wav_path.exists():
                    return str(wav_path.absolute())
            except Exception:
                pass
        
        # fallback to pydub (requires ffmpeg)
        if PYDUB_AVAILABLE:
            try:
                wav_path = audio_path.parent / f"{audio_path.stem}_converted.wav"
                wav_path = wav_path.resolve()
                
                audio = AudioSegment.from_file(str(audio_path))
                audio.export(str(wav_path), format="wav")
                
                if wav_path.exists():
                    return str(wav_path.absolute())
            except Exception:
                pass
        
        # if all conversions fail, return original (whisper might handle it)
        return str(audio_path.absolute())
    
    def transcribe(
        self, 
        file_path: Union[str, Path], 
        output_file: Optional[Union[str, Path]] = None
    ) -> str:
        # transcribe audio or video file to text
        # args: file_path - path to audio (mp3) or video (mp4) file
        #       output_file - optional path to save transcription text file
        # returns: transcribed text as string
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")
        
        # load model if needed
        self._load_model()
        
        # handle video files - extract audio first
        audio_path = file_path
        temp_audio = None
        converted_audio = None
        
        if self._is_video_file(file_path):
            audio_path = Path(self._extract_audio_from_video(file_path))
            temp_audio = str(audio_path)
            # verify extracted audio exists
            if not audio_path.exists():
                raise FileNotFoundError(f"extracted audio file not found: {audio_path}")
        
        try:
            # load audio as numpy array to avoid ffmpeg dependency
            audio_array = None
            sample_rate = 16000
            
            if LIBROSA_AVAILABLE:
                try:
                    # load audio with librosa (no ffmpeg needed)
                    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
                except Exception as e:
                    # if librosa fails, try converting to wav first
                    audio_path = Path(self._convert_audio_to_wav(audio_path))
                    if str(audio_path) != str(file_path) and audio_path.suffix == '.wav' and '_converted' in audio_path.stem:
                        converted_audio = str(audio_path)
                    audio_array, sample_rate = librosa.load(str(audio_path), sr=16000)
            
            if audio_array is not None:
                # transcribe using numpy array (avoids ffmpeg)
                result = self.model.transcribe(audio_array)
            else:
                # fallback to file path (requires ffmpeg)
                audio_path_str = str(audio_path.absolute())
                if not audio_path.exists():
                    raise FileNotFoundError(f"audio file not found: {audio_path}")
                result = self.model.transcribe(audio_path_str)
            
            text = result["text"].strip()
            
            # save to file if output path provided
            if output_file:
                output_path = Path(output_file).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
            
            return text
        
        finally:
            # cleanup temporary audio files
            for temp_file in [temp_audio, converted_audio]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

