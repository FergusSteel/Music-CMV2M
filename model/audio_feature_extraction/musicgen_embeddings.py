import torch
import torch.nn as nn
import torchaudio
from transformers import AutoTokenizer, AutoModel, EncodecModel
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torch.optim as optim
import os
import glob

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
    

# FEASABILITY STUDY - Decoder
# class UNetAudioDecoder(nn.Module):
#     def __init__(self, embedding_dim, output_length):
#         super(UNetAudioDecoder, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.output_length = output_length

#         self.encoder = nn.Sequential(
#             nn.Conv1d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),  # Down-sample
#             nn.ReLU(),
#             nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(512, 1024, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         self.bottleneck = nn.Sequential(
#             nn.Conv1d(1024, 1024, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(1024, 512, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose1d(256, 1, kernel_size=4, stride=2, padding=1),  # Reconstruct single-channel audio
#         )

#     def forward(self, embeddings):
#         x = embeddings.transpose(1, 2)  # Shape: [batch_size, embedding_dim, seq_len]
        
#         enc_out = self.encoder(x)

#         bottleneck_out = self.bottleneck(enc_out)

#         reconstructed_audio = self.decoder(bottleneck_out)
#         reconstructed_audio = reconstructed_audio.view(reconstructed_audio.size(0), -1)  # [batch_size, output_length]
#         return reconstructed_audio



# # Initialize feature extractor and decoder
# if __name__ == "__main__":
#     feature_extractor = AudioFeatureExtraction()
#     decoder = UNetAudioDecoder(embedding_dim=1024, output_length=320000).to("cuda")  # Adjust output_length for your audio duration

#     # Loss function and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

#     # Training loop
#     def train_unet(audio_files, decoder, feature_extractor, epochs=1, save_path="weights.pth"):
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(decoder.parameters(), lr=1e-3)

#         for epoch in range(epochs):
#             for audio_path in audio_files:
#                 # Extract features
#                 embeddings = feature_extractor.extract_features(audio_path).mean(dim=1).to("cuda")  # Average over sequence
                
#                 # Load original audio
#                 waveform, sample_rate = torchaudio.load(audio_path)
#                 waveform = waveform.to("cuda")
                
#                 waveform = waveform[:, :320000]
#                 if waveform.size(1) < 320000:
#                     waveform = torch.cat([waveform, torch.zeros(1, 320000 - waveform.size(1)).to("cuda")], dim=1)

#                 # Forward pass
#                 optimizer.zero_grad()
#                 reconstructed_audio = decoder(embeddings)

#                 # Compute loss
#                 loss = criterion(reconstructed_audio, waveform)
#                 loss.backward()
#                 optimizer.step()

#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

#         # Save trained weights
#         torch.save(decoder.state_dict(), save_path)
#         print(f"Model weights saved to {save_path}")

def collect_audio_files(directory="../../Solos/processed_audios/Cello", extension="mp3"):
    return glob.glob(f"{directory}/*.{extension}")

audio_dir = "../../Solos/processed_audios/Cello"
audio_files = collect_audio_files(audio_dir, extension="mp3")

model = AudioFeatureExtraction()

extracted = model.extract_features(audio_files[0])
