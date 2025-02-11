import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer, AutoModel, EncodecModel, AutoProcessor

class VideoToAudioGenerator(nn.Module):
    def __init__(self, feature_dim=768, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        
        self.video_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.encodec_predictor = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2000),
            nn.GELU()
        )
        
        self.to(self.device)
        
    def forward(self, video_features):
        latent = self.video_encoder(video_features)
        shared_features = self.encodec_predictor(latent)

        print("post predictor", shared_features.shape)
        
        batch_size = video_features.shape[0]
        nb_chunks = video_features.shape[1]
        
        audio_codes = shared_features.reshape(batch_size, 1, 4, 500).long()
        audio_scales = torch.ones((batch_size, nb_chunks), device=self.device)
        
        return {
            'latent': latent,
            'audio_codes': audio_codes,
            'audio_scales': audio_scales
        }
    
    def compute_loss(self, generated, target_codes):
        code_loss = F.mse_loss(generated['audio_codes'], target_codes)
        return {'total_loss': code_loss}

def generate_audio(model, encodec_model, video_features, padding_mask=None):
    with torch.no_grad():
        outputs = model(video_features)
        
        audio_values = encodec_model.decode(
            audio_codes=outputs['audio_codes'],
            audio_scales=outputs['audio_scales'],
            padding_mask=padding_mask
        ).audio_values
        
        return audio_values

def train_step(model, video_features, target_codes, optimizer):
    generated = model(video_features)
    losses = model.compute_loss(generated, target_codes)
    
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()
    
    return losses
    
if __name__ == "__main__":
    # Test the model with some sample data
    import os
    from feature_extraction import MultiModalFeatureExtractor
    import sounddevice as sd
    
    # Initialize models
    generator = VideoToAudioGenerator()
    extractor = MultiModalFeatureExtractor()
    
    # Load a test video-audio pair
    audio_directory = "../Solos/processed_audios/Cello"
    video_directory = "../Solos/processed_videos/Cello"
    
    audio_path = os.path.join(audio_directory, "-qRn8UyHogA_segment_1.mp3")
    video_path = os.path.join(video_directory, "-qRn8UyHogA.f136_segment_1.mp4")
    
    # Extract features
    features = extractor.extract_features(video_path, audio_path)
    
    features["video_features"] = features["video_features"].to(generator.device)
    print("Devices for Generator and Features:", generator.device, features["video_features"].device)
    outputs = generator(features["video_features"])
    
    print("Generated Codes Shape:", outputs["audio_codes"].shape)
    print("Original Encodec Codes Shape:", features["encodec_features"].shape)


    # encodec decocer
    # Try to reconstruct audio using Encodec
    with torch.no_grad():
        audio_values = extractor.audio_extractor.model.decode(
            audio_codes=outputs["audio_codes"],
            audio_scales =outputs["audio_scales"],
            padding_mask=features["additional_info"]["padding_mask"],
        ).audio_values
    
    # Play both original and generated audio
    print("\nPlaying original audio...")
    original_waveform = extractor.audio_extractor._preprocess_audio(audio_path)
    sd.play(original_waveform.cpu().numpy(), 32000)
    sd.wait()
    
    print("\nPlaying generated audio...")
    sd.play(audio_values.cpu().numpy(), 32000)
    sd.wait()