import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from transformers import EncodecModel
import torchaudio
from transformers import AutoTokenizer, AutoModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch.optim as optim
import torchaudio.transforms as T
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

os.environ["HF_HOME"] = "E:/HuggingFace/huggingface_cache"

class TemporalAudioEncoder:
    def __init__(self, model_name="facebook/encodec_32khz", target_length=1568):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncodecModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.target_length = target_length
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=2048,
            hop_length=160,
            n_mels=128,
            power=2.0
        ).to(self.device)
        
        self.projection = nn.Linear(128, 768).to(self.device)
        
    def extract_features(self, audio_path):
        waveform = self._preprocess_audio(audio_path)
        original_length = waveform.size(-1)
        
        inputs = self.processor(raw_audio=waveform, sampling_rate=32000, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            encodec_output = self.model.encode(inputs["input_values"])
            self.last_audio_codes = encodec_output.audio_codes
            batch_size, _, _, nb_chunks = self.last_audio_codes.shape
            self.last_audio_scales = torch.ones((batch_size, nb_chunks), device=self.device)
            
        mel_features = self.mel_spec(waveform.to(self.device))
        if mel_features.dim() == 2:
            mel_features = mel_features.unsqueeze(0)
            
        self.original_time_dim = mel_features.size(2)
        
        aligned_features = self._align_temporal_dimension(mel_features)
        projected_features = self.projection(aligned_features)

        return {
            "encodec_features": self.last_audio_codes,
            "audio_scales": self.last_audio_scales,
            "mel_features": mel_features,
            "aligned_features": projected_features,
            "original_length": original_length,
            "padding_mask": inputs.get("padding_mask", None)
        }
    
    def reconstruct_from_encodec(self):
        """Reconstruct audio using EncodecModel's decoder"""
        with torch.no_grad():
            reconstructed = self.model.decode(
                audio_codes=self.last_audio_codes,
                audio_scales=self.last_audio_scales
            ).audio_values
        return reconstructed.squeeze()
    
    def _preprocess_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if sample_rate != self.processor.sampling_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.processor.sampling_rate
            )(waveform)
        
        return waveform.squeeze(0)
    
    def _align_temporal_dimension(self, features):
        features = features.permute(0, 2, 1)  # [batch, time, mels]
        
        features = F.interpolate(
            features.unsqueeze(1),
            size=(self.target_length, features.size(-1)),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return features
    
    def visualize_features(self, features):
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        mel_spec = features["mel_features"].squeeze().cpu().numpy()
        axes[0].imshow(mel_spec, aspect='auto', origin='lower')
        axes[0].set_title("Mel Spectrogram")
        
        aligned = features["aligned_features"].squeeze().mean(-1).cpu().numpy()
        axes[1].imshow(aligned.reshape(1, -1), aspect='auto')
        axes[1].set_title("Aligned Features (Mean)")
        
        encodec = features["encodec_features"].squeeze().cpu().numpy()
        axes[2].imshow(encodec, aspect='auto', origin='lower')
        axes[2].set_title("Encodec Features")
        
        plt.tight_layout()
        return fig

def test_encoder(audio_path):
    """Test the encoding and feature visualization"""
    encoder = TemporalAudioEncoder()
    
    features = encoder.extract_features(audio_path)
    print(f"Encodec features shape: {features['encodec_features'].shape}")
    print(f"Mel features shape: {features['mel_features'].shape}")
    print(f"Aligned features shape: {features['aligned_features'].shape}")
    
    encoder.visualize_features(features)
    
    reconstructed = encoder.reconstruct_from_encodec()
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    return features, reconstructed

def collect_audio_files(directory="../../Solos/processed_audios/Cello", extension="mp3"):
    return glob.glob(f"{directory}/*.{extension}")

# Test encodecModel decoder
# if __name__ == "__main__":
#     audio_files = collect_audio_files()
#     audio_path = audio_files[0]
    
#     features, reconstructed = test_encoder(audio_path)
    
#     original = torchaudio.load(audio_path)[0].squeeze()
#     print("\nAudio playback:")
#     print("Original:")
#     sd.play(original.numpy(), 32000)
#     sd.wait()
    
#     print("Reconstructed:")
#     sd.play(reconstructed.numpy(), 32000)
#     sd.wait()
    
#     plt.show()
