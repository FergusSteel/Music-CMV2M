import torch
import torch.nn as nn
import torchaudio
from transformers import AutoTokenizer, AutoModel, EncodecModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch.optim as optim
import torchaudio.transforms as T
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

os.environ["HF_HOME"] = "E:/HuggingFace/huggingface_cache"

class AudioFeatureExtraction:
    def __init__(self, model_name="facebook/encodec_32khz", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Load MusicGens pretrained tokenizer uno
        self.model =  EncodecModel.from_pretrained("facebook/encodec_32khz").to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")
        self.model.eval()

    def extract_features(self, audio_path):
        audio_sample = self._preprocess_audio(audio_path)

        audio_sample = audio_sample

        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.processor.sampling_rate, return_tensors="pt").to(self.device)

        # tokenize that audio man
        # with torch.no_grad():
        #     encoded = self.model.encode(inputs["input_values"].to(self.device))
        #     tokens = encoded.audio_codes
        # print(tokens.shape)
        audio_codes = self.model(**inputs).audio_codes
        return audio_codes
        


    def _preprocess_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        print(sample_rate)
        print(waveform.shape)
        if sample_rate != self.processor.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.processor.sampling_rate)(waveform)
        print(waveform.shape)
        return waveform.squeeze(0)
    

class AudioSpectogramEncoder:
    def __init__(self, sample_rate=32000, n_fft=2048, hop_length=203, n_mels=128, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def waveform_to_mel(self, waveform):
        # Convert waveform to Mel-spectrogram
        print("waveform", waveform.shape)
        mel_spec = self.mel_spectrogram(waveform)  # Shape: [n_mels, time_frames]
        return mel_spec
    
    def _preprocess_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        print(sample_rate)
        print(waveform.shape)
        if sample_rate != self.processor.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.processor.sampling_rate)(waveform)
        print(waveform.shape)
        return waveform.squeeze(0)
    
class UniformChunkSpectrogram:
    def __init__(self, num_chunks=1568, n_fft=2048, n_mels=128, sample_rate=32000):
        self.num_chunks = num_chunks
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel_filter = T.MelScale(n_mels=n_mels, sample_rate=sample_rate)

    def chunk_and_transform(self, waveform):
        total_samples = waveform.size(-1)
        chunk_size = total_samples // self.num_chunks

        spectrogram = []

        for i in range(self.num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = waveform[..., start:end]

            # Ensure the chunk has enough length for n_fft
            if chunk.size(-1) < self.n_fft:
                chunk = torch.nn.functional.pad(chunk, (0, self.n_fft - chunk.size(-1)))

            # Apply STFT
            chunk_fft = torch.stft(chunk, n_fft=self.n_fft, return_complex=True)
            chunk_power = torch.abs(chunk_fft).pow(2)

            # Convert to Mel-scale
            mel_spec = self.mel_filter(chunk_power)
            
            # Aggregate across frequency dimensions
            mel_spec = mel_spec.mean(dim=-1)  # Average over frequencies
            
            spectrogram.append(mel_spec)

        # Stack spectrogram chunks into [num_chunks, n_mels]
        return torch.stack(spectrogram, dim=0)
    
class FixedSizeMelSpectrogram:
    def __init__(self, sample_rate=32000, n_fft=2048, hop_length=203, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def waveform_to_mel(self, waveform):
        # Ensure waveform has the correct shape
        waveform = waveform.squeeze(0)  # Remove batch dimension

        # Compute Mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # Shape: [n_mels, time_frames]

        # Check temporal dimensions and trim/pad if necessary
        num_frames = mel_spec.size(-1)
        if num_frames < 1568:
            # Pad if fewer frames
            mel_spec = F.pad(mel_spec, (0, 1568 - num_frames))
        elif num_frames > 1568:
            # Trim if more frames
            mel_spec = mel_spec[..., :1568]

        return mel_spec

def collect_audio_files(directory="../../Solos/processed_audios/Cello", extension="mp3"):
    return glob.glob(f"{directory}/*.{extension}")

audio_dir = "../../Solos/processed_audios/Violin"
audio_files = collect_audio_files(audio_dir, extension="mp3")

model = AudioFeatureExtraction()

specto = AudioSpectogramEncoder()

# specto2 = UniformChunkSpectrogram()

# specogram2 = specto2.chunk_and_transform(model._preprocess_audio(audio_files[0]))
# print("specogram2", specogram2.shape)

spectrogram_extractor = FixedSizeMelSpectrogram()

def visualize_mel_spectrogram(mel_spec, sample_rate=32000, hop_length=203, n_mels=128):
    # Convert to numpy array for visualization
    mel_spec = mel_spec.cpu().detach().numpy() if isinstance(mel_spec, torch.Tensor) else mel_spec

    # Generate time and frequency axes
    time_axis = np.linspace(0, mel_spec.shape[1] * hop_length / sample_rate, mel_spec.shape[1])
    freq_axis = np.linspace(0, sample_rate / 2, n_mels)

    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis', 
                extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Mel-Spectrogram")
    plt.show()
try:
    for f in audio_files:
        print(f)
        mel_spec = spectrogram_extractor.waveform_to_mel(model._preprocess_audio(f))
        print(f"Mel-Spectrogram Shape: {mel_spec.shape}")
        visualize_mel_spectrogram(mel_spec)
except Exception as e:
    print(f"Error: {e}")


specogram = specto.waveform_to_mel(model._preprocess_audio(audio_files[5]))
print("spectogram", specogram.shape)
extracted = model.extract_features(audio_files[0])


