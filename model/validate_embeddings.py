import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from audio_feature_extraction import TemporalAudioEncoder
import sounddevice as sd

def demo_audio_encoder(audio_path, encoder):
    features = encoder.extract_features(audio_path)
    print(f"Encodec features shape: {features['encodec_features'].shape}")
    print(f"Mel features shape: {features['mel_features'].shape}")
    print(f"Aligned features shape: {features['aligned_features'].shape}")
    
    reconstructed = encoder.reconstruct_from_encodec()
    print(f"Reconstructed audio shape: {reconstructed.shape}")
    
    batch_size = 4
    dummy_batch = features["aligned_features"].repeat(batch_size, 1, 1)
    
    return features, reconstructed

def visualize_features(features):
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    mel_features = features["mel_features"].detach().cpu()
    axes[0].imshow(mel_features.squeeze().numpy(), 
                  aspect='auto', origin='lower')
    axes[0].set_title("Mel Spectrogram")
    
    aligned_features = features["aligned_features"].detach().cpu()
    axes[1].imshow(aligned_features.squeeze().mean(-1).numpy().reshape(1, -1), 
                  aspect='auto')
    axes[1].set_title("Aligned Features (Mean)")
    
    encodec_features = features["encodec_features"].detach().cpu()
    axes[2].imshow(encodec_features.squeeze().numpy(), 
                  aspect='auto', origin='lower')
    axes[2].set_title("Encodec Features")
    
    plt.tight_layout()
    plt.show()

def play_audio(waveform, sample_rate=32000):
    print("Press Enter to play audio...")
    input()
    sd.play(waveform.numpy(), sample_rate)
    sd.wait()

if __name__ == "__main__":
    encoder = TemporalAudioEncoder(target_length=1568)
    audio_path = "../../Solos/processed_audios/Violin/0dKj8oP3QxE_segment_1.mp3"
    
    features, reconstructed = demo_audio_encoder(audio_path, encoder)
    
    visualize_features(features)
    
    original = encoder._preprocess_audio(audio_path)
    reconstructed = reconstructed.cpu()
    
    print("\nAudio playback:")
    print("Original:")
    play_audio(original)
    
    print("Reconstructed:")
    play_audio(reconstructed)
    
    min_length = min(original.size(-1), reconstructed.size(-1))
    original = original[:min_length]
    reconstructed = reconstructed[:min_length]
    
    mse = F.mse_loss(original, reconstructed)
    print(f"\nReconstruction Quality Metrics:")
    print(f"MSE: {mse.item():.6f}")
    
    print("\nFeature shapes:")
    print(f"Mel features: {features['mel_features'].shape}")
    print(f"Aligned features: {features['aligned_features'].shape}")
    print(f"Encodec features: {features['encodec_features'].shape}")