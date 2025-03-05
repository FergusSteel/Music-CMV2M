import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import re
import numpy as np

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

class FeatureExtractionDataset(Dataset):
    def __init__(self, data_dir, instrument_type="Violin", feature_extractor=None, feature_cache_dir=None):
        self.video_dir = os.path.join(data_dir, "processed_videos", instrument_type)
        print(self.video_dir)
        self.audio_dir = os.path.join(data_dir, "processed_audios", instrument_type)
        self.feature_extractor = feature_extractor
        self.feature_cache_dir = feature_cache_dir
        
        if feature_cache_dir:
            # Create separate directories for each modality
            self.modality_dirs = {
                'video_features': os.path.join(feature_cache_dir, 'video'),
                'spectogram': os.path.join(feature_cache_dir, 'spectogram'),
                'optical_flow': os.path.join(feature_cache_dir, 'optical_flow'),
                'encodec_features': os.path.join(feature_cache_dir, 'encodec')
            }
            for dir_path in self.modality_dirs.values():
                os.makedirs(dir_path, exist_ok=True)
        
        self.video_files = sorted(glob.glob(os.path.join(self.video_dir, "*.mp4")))
        self.paired_files = []

        print(self.video_files)
        
        for video_path in self.video_files:
            
            video_name = os.path.basename(video_path)
            base_name = re.sub(r'\.f\d+', '', video_name)
            audio_name = base_name[:-4] + ".mp3"
            audio_path = os.path.join(self.audio_dir, audio_name)
            
            print(video_name, audio_name)

            if os.path.exists(audio_path):
                self.paired_files.append((video_path, audio_path))
        
        print(f"Found {len(self.paired_files)} paired audio-video segments")

    def __len__(self):
        return len(self.paired_files)
    
    def __getitem__(self, idx):
        video_path, audio_path = self.paired_files[idx]
        base_name = os.path.basename(video_path)[:-4]  # Remove .mp4        
        # Check if all features are cached
        if self.feature_cache_dir:
            cached_features = {}
            all_cached = True
            
            for modality, cache_dir in self.modality_dirs.items():
                cache_path = os.path.join(cache_dir, f"{base_name}.pt")
                if os.path.exists(cache_path):
                    cached_features[modality] = torch.load(cache_path)
                else:
                    all_cached = False
                    break
            
            if all_cached:
                return cached_features
        
        # get features if not cached
        if self.feature_extractor:
            features = self.feature_extractor.extract_features(video_path, audio_path)
            
            if self.feature_cache_dir:
                for modality, tensor in features.items():
                    if modality in self.modality_dirs:
                        cache_path = os.path.join(
                            self.modality_dirs[modality], 
                            f"{base_name}.pt"
                        )
                        torch.save(tensor, cache_path)
            
            return features
        else:
            return {
                'video_path': video_path,
                'audio_path': audio_path
            }

def create_modal_specific_datasets(feature_dataset, batch_size=8):
    full_data = []
    for idx in range(len(feature_dataset)):
        features = feature_dataset[idx]
        full_data.append(features)
    
    # Create datasets for each modality
    video_features = [item["video_features"] for item in full_data]
    optical_flow = [item["optical_flow"] for item in full_data]
    spectogram = [item["spectogram"] for item in full_data]
    encodec_features = [item["encodec_features"] for item in full_data]

    print("length spector", len(spectogram))
    
    # Create data loaders
    video_loader = DataLoader(video_features, batch_size=batch_size, shuffle=True)
    flow_loader = DataLoader(optical_flow, batch_size=batch_size, shuffle=True)
    spectogram_loader = DataLoader(spectogram, batch_size=batch_size, shuffle=True)
    encodec_loader = DataLoader(encodec_features, batch_size=batch_size, shuffle=True)
    
    return {
        "video": (video_loader, DataLoader(video_features, batch_size=batch_size, shuffle=False)),
        "optical_flow": (flow_loader, DataLoader(optical_flow, batch_size=batch_size, shuffle=False)),
        "spectogram": (spectogram_loader, DataLoader(spectogram, batch_size=batch_size, shuffle=False)),
        "encodec": (encodec_loader, DataLoader(encodec_features, batch_size=batch_size, shuffle=False))
    }

def extract_and_save_features(data_dir, instrument_type, feature_extractor, output_dir):
    dataset = FeatureExtractionDataset(
        data_dir, 
        instrument_type,
        feature_extractor=feature_extractor,
        feature_cache_dir=output_dir
    )

    print(len(dataset))
    
    for idx in range(len(dataset)):
        _ = dataset[idx]  # This will extract and cache features
    
    print(f"Extracted features for {len(dataset)} pairs")

def create_training_loaders(feature_dir, modality, batch_size=8, train_split=0.8):
    print("Feature dir", feature_dir)
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
    
    modality_data = []
    for file_path in feature_files:
        features = torch.load(file_path)
        if modality == "video":
            modality_data.append(features)
        elif modality == "optical_flow":
            modality_data.append(features)
        elif modality == "spectogram":
            modality_data.append(features)
        elif modality == "encodec":
            modality_data.append(features)
    
    indices = list(range(len(modality_data)))
    np.random.shuffle(indices)
    split = int(train_split * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_data = [modality_data[i] for i in train_idx]
    val_data = [modality_data[i] for i in val_idx]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader