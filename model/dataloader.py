import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import re

class AudioVideoDataset(Dataset):
    def __init__(self, data_dir, instrument_type="Violin"):
        """
        Args:
            data_dir: Root directory containing processed_videos and processed_audios
            instrument_type: Type of instrument (subfolder name)
        """
        self.video_dir = os.path.join(data_dir, "processed_videos", instrument_type)
        self.audio_dir = os.path.join(data_dir, "processed_audios", instrument_type)
        
        # Get all video files
        self.video_files = sorted(glob.glob(os.path.join(self.video_dir, "*_segment_*.mp4")))
        self.paired_files = []
        
        for video_path in self.video_files:
            video_name = os.path.basename(video_path)
            base_name = re.sub(r'\.f\d+', '', video_name)  # Remove that thing that like .fXXX
            audio_name = base_name[:-4] + ".mp3"
            audio_path = os.path.join(self.audio_dir, audio_name)
            
            if os.path.exists(audio_path):
                self.paired_files.append((video_path, audio_path))
                print(f"Found matching pair: {video_name} -> {audio_name}")
        
        print(f"Found {len(self.paired_files)} paired audio-video segments")
        
        if len(self.paired_files) == 0:
            raise ValueError("No paired files found! Check the paths and file naming convention.")

    def __len__(self):
        return len(self.paired_files)
    
    def __getitem__(self, idx):
        video_path, audio_path = self.paired_files[idx]
        return {
            'video_path': video_path,
            'audio_path': audio_path
        }

def create_data_loaders(data_dir, instrument_type="Violin", batch_size=8, train_split=0.8):
    """Create train and validation data loaders"""
    dataset = AudioVideoDataset(data_dir, instrument_type)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    data_dir = "../Solos"
    train_loader, val_loader = create_data_loaders(data_dir, "Violin")
    
    # Test loading a batch
    for batch in train_loader:
        print("Video paths:", batch['video_path'])
        print("Audio paths:", batch['audio_path'])
        break