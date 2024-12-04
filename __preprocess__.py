import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
import librosa
import cv2

class CrossModalDataset(Dataset):
    def __init__(self, video_dir, audio_dir, target_video_size=(224, 224), target_fps=25, target_audio_rate=44100, duration=None):
        """
        A Dataset class for loading and preprocessing video and audio pairs.

        Args:
            video_dir (str): Directory containing video files.
            audio_dir (str): Directory containing audio files.
            target_video_size (tuple): Target size for video frames (H, W).
            target_fps (int): Target frames per second for videos.
            target_audio_rate (int): Target sampling rate for audio.
            duration (float): Max duration (in seconds) for video/audio.
        """
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.target_video_size = target_video_size
        self.target_fps = target_fps
        self.target_audio_rate = target_audio_rate
        self.duration = duration

        # List all video and audio files
        self.video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".mp3")])

        # Check if video and audio are aligned
        self._validate_data()

        # Video transformation pipeline
        self.video_transforms = Compose([
            Resize(target_video_size),
            CenterCrop(target_video_size),
            ToTensor()
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])

        # Load video and audio
        video_frames = self._load_video(video_path)
        audio_data, audio_rate = self._load_audio(audio_path)

        # Ensure duration alignment
        video_frames, audio_data = self._align_duration(video_frames, audio_data, audio_rate)

        return {
            "video": video_frames,  # Shape: [num_frames, 3, H, W]
            "audio": audio_data,    # Shape: [num_samples]
            "video_file": self.video_files[idx],
            "audio_file": self.audio_files[idx]
        }

    def _validate_data(self):
        """Ensure each video has a corresponding audio file."""
        video_basenames = [os.path.splitext(f)[0] for f in self.video_files]
        audio_basenames = [os.path.splitext(f)[0] for f in self.audio_files]
        if video_basenames != audio_basenames:
            raise ValueError("Video and audio files are not aligned!")

    def _load_video(self, video_path):
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to RGB and apply transformations
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.video_transforms(frame)
            frames.append(frame)
        cap.release()

        # Sample frames to target FPS
        sampled_frames = self._sample_frames(frames, fps)
        return torch.stack(sampled_frames)

    def _sample_frames(self, frames, fps):
        """Sample frames to match target FPS."""
        interval = int(fps / self.target_fps)
        return frames[::interval]

    def _load_audio(self, audio_path):
        """Load and preprocess audio data."""
        audio, rate = librosa.load(audio_path, sr=None)
        if rate != self.target_audio_rate:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=self.target_audio_rate)
        return audio, self.target_audio_rate

    def _align_duration(self, video_frames, audio_data, audio_rate):
        """Ensure video and audio have the same duration."""
        video_duration = len(video_frames) / self.target_fps
        audio_duration = len(audio_data) / audio_rate

        if self.duration:
            max_duration = self.duration
        else:
            max_duration = min(video_duration, audio_duration)

        # Trim/pad video
        num_frames = int(max_duration * self.target_fps)
        if len(video_frames) > num_frames:
            video_frames = video_frames[:num_frames]
        elif len(video_frames) < num_frames:
            padding = torch.zeros((num_frames - len(video_frames), *video_frames.shape[1:]))
            video_frames = torch.cat([video_frames, padding])

        # Trim/pad audio
        num_samples = int(max_duration * audio_rate)
        if len(audio_data) > num_samples:
            audio_data = audio_data[:num_samples]
        elif len(audio_data) < num_samples:
            audio_data = librosa.util.fix_length(audio_data, size=num_samples)

        return video_frames, audio_data
