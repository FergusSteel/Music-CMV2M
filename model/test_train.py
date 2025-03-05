import os
import shutil
import random
import glob
import re

def create_test_dataset(source_dir, dest_dir, instrument="Violin", num_samples=20):
    """
    Copy a small subset of paired videos and audios for testing
    """
    # Setup source paths
    video_src = os.path.join(source_dir, "processed_videos", instrument)
    audio_src = os.path.join(source_dir, "processed_audios", instrument)
    
    # Setup destination paths
    video_dest = os.path.join(dest_dir, "processed_videos", instrument)
    audio_dest = os.path.join(dest_dir, "processed_audios", instrument)
    os.makedirs(video_dest, exist_ok=True)
    os.makedirs(audio_dest, exist_ok=True)
    
    # Get all video files
    video_files = glob.glob(os.path.join(video_src, "*_segment_*.mp4"))
    print(len(video_files))
    
    # Randomly select num_samples files
    selected_videos = random.sample(video_files, min(num_samples, len(video_files)))
    print(len(selected_videos))
    
    # Copy each selected video and its corresponding audio
    copied_pairs = 0
    for video_path in selected_videos:
        video_name = os.path.basename(video_path)
        # Remove .fXXX and .mp4, keep the segment part
        base_name = re.sub(r'\.f\d+', '', video_name[:-4])
        audio_name = f"{base_name}.mp3"
        
        print(video_name, audio_name)

        audio_path = os.path.join(audio_src, audio_name)
        
        if os.path.exists(audio_path):
            # Copy files
            shutil.copy2(video_path, os.path.join(video_dest, video_name))
            shutil.copy2(audio_path, os.path.join(audio_dest, audio_name))
            copied_pairs += 1
            print(f"Copied pair {copied_pairs}: {video_name} -> {audio_name}")
    
    print(f"\nCreated test dataset with {copied_pairs} pairs")
    return copied_pairs

# if __name__ == "__main__":
#     # Set paths
#     source_dir = "../dat/"
#     test_dir = "../test_data"  # Where to put the test dataset
#
#     # Create test dataset
#     num_pairs = create_test_dataset(
#         source_dir=source_dir,
#         dest_dir=test_dir,
#         instrument="Violin",
#         num_samples=20
#     )

import torch
from feature_extraction import MultiModalFeatureExtractor
from feature_encoders import (VideoFeatureEncoder, OpticalFlowEncoder,
                            EncodecEncoder, SpectrogramEncoder,
                            OpticalFlowDecoder, EncodecDecoder, SpectrogramDecoder, EncoderTrainer)
from dataloader import (FeatureExtractionDataset, extract_and_save_features, create_training_loaders)

def test_feature_extraction():
    """Test feature extraction and caching"""
    print("\n=== Testing Feature Extraction ===")
    
    # Initialize feature extractor
    extractor = MultiModalFeatureExtractor()
    
    # Extract and cache features
    extract_and_save_features(
        data_dir="..\\test_data",
        instrument_type="Violin",
        feature_extractor=extractor,
        output_dir="..\\test_features"
    )

def test_encoder_training(modality, encoder, decoder, batch_size=4):
    """Test training for a specific encoder"""
    print(f"\n=== Testing {modality} Encoder Training ===")

    # Create data loaders


    # Get a sample batch to check shapes
    for batch in train_loader:
        print(f"Sample batch shape: {batch.shape}")
        break

    # Test one epoch of training
    print("\nTesting one training epoch...")
    trainer = EncoderTrainer(encoder, decoder)

    epoch_loss = 0
    num_batches = 0

    for batch in train_loader:
        loss = trainer.train_step(batch)
        epoch_loss += loss
        num_batches += 1
        print(f"Batch {num_batches} Loss: {loss:.4f}")

    print(f"Average Training Loss: {epoch_loss/num_batches:.4f}")

    # Test validation
    print("\nTesting validation...")
    val_loss = trainer.validate(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

def main():
    print("Starting pipeline test with test_data...")

    # First extract and cache features
    test_feature_extraction()

    # Test Optical Flow encoder
    flow_encoder = OpticalFlowEncoder()
    flow_decoder = OpticalFlowDecoder()
    test_encoder_training("optical_flow", flow_encoder, flow_decoder)

    # Test Encodec encoder
    encodec_encoder = EncodecEncoder()
    encodec_decoder = EncodecDecoder()
    test_encoder_training("encodec", encodec_encoder, encodec_decoder)

    # Test Spectrogram encoder
    spec_encoder = SpectrogramEncoder()
    spec_decoder = SpectrogramDecoder()
    test_encoder_training("spectogram", spec_encoder, spec_decoder)

if __name__ == "__main__":
    main()