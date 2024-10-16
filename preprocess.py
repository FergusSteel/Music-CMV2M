import os
import torch
from torch.utils.data import DataLoader, Dataset
import librosa
import torchvision.transforms as T
import cv2

class AudioVisualDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        # this gets all da videos
        self.video_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.mp4') and file.startswith('Vid'):
                    self.video_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        # This needs to change as each video will have several audio files (one for each instrument)
        audio_path = video_path.replace('.mp4', '.wav').replace("Vid", "AuMix")

        # Load video frames and audio
        video_data = self.load_video(video_path)
        audio_data = self.load_audio(audio_path)

        return video_data, audio_data

    def load_video(self, video_path, n=2):
        video_capture = cv2.VideoCapture(video_path)

        frames = []
        success, frame = video_capture.read()

        while success:
            # Convert the frame from BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # This is where I need to crop X number of 224x224 frames from the video where X is nyumber of audio files to corresponding videos
            frame_rgb = cv2.resize(frame_rgb, (224, 224))  # resize to 224x224 for ViT

            frame_tensor = T.ToTensor()(frame_rgb)
            frames.append(frame_tensor)
            success, frame = video_capture.read()

        video_capture.release()

        # put the frames into a single tensor - [num_frames, 3, 224, 224]
        video_tensor = torch.stack(frames)

        return video_tensor

    def load_audio(self, audio_path):
        # Load audio using librosa and convert it to a spectrogram
        y, sr = librosa.load(audio_path, sr=None)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        return spectrogram

def preprocess_data(dataset_path, batch_size):
    dataset = AudioVisualDataset(dataset_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
