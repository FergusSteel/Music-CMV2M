import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_encoders import VideoFeatureEncoder, OpticalFlowEncoder, EncodecEncoder,SpectogramEncoder

class SharedLatentSpace(nn.Module):
    def __init__(self, feature_dim=768, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.temporal_alignment = TemporalCrossAlignmentModule().to(self.device)

        # REPLACE THESE WITH feature_encoders

        # Project the videomae features to latent space man WE NEED TO ADD IN THE OPTICAL FLOW
        self.video_encoder = VideoFeatureEncoder()

        self.optical_flow_encoder = OpticalFlowEncoder()

        # Encodec token encoder
        self.encodec_encoder = EncodecEncoder()
        
        #Projectin audio to latent space 
        self.audio_encoder = SpectogramEncoder()

        
        # Temperature parameter for contrastive loss
        self.token_embedding = nn.Embedding(2048, latent_dim).to(self.device)
        self.token_proj = nn.Linear(latent_dim, latent_dim).to(self.device)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        self.to(self.device)
    
    def encode_video(self, video_features, optical_flow):
        flow_features = self.optical_flow_encoder(optical_flow)
        video_latent = self.video_encoder(video_features)

        # May fuse features differently
        combined_features = video_latent + flow_features
        
        return combined_features
    
    def encode_audio(self, audio_features, encodec_features):
        # Encode aligned features
        spectogram_features = self.audio_encoder(audio_features)
        encodec_features = self.encodec_encoder(encodec_features)

        # this should maybe be a concat job
        audio_latent = spectogram_features + encodec_features
        
        return audio_latent
    
    def align_modalities(self, video_latent, audio_latent):
        fused_video, fused_audio, video_attention, audio_attention = self.temporal_alignment(video_latent, audio_latent)

        return fused_video, fused_audio, video_attention, audio_attention
        
    def encode(self, features):
            # Encode both modalities onto latent space and align em
        print("Video Features Shape: ", features["video_features"].shape)
        print("Audio Features Shape: ", features["audio_spectograms"].shape)
        print("Optical Flow Shape: ", features["optical_flow"].shape)
        print("Encodec Features Shape: ", features["encodec_features"].shape)

        video_latent = self.encode_video(
            features["video_features"],
            features["optical_flow"])
        
        audio_latent = self.encode_audio(
            features["audio_spectograms"],
            features["encodec_features"]
        )

        fused_video, fused_audio, video_attention, audio_attention = self.align_modalities(video_latent, audio_latent)
        
        return {
            "video_embedding": fused_video,     
            "audio_embedding": fused_audio,      
            "video_attention_map": video_attention,
            "audio_attention_map": audio_attention,          
            "video_latent": video_latent,       
            "audio_latent": audio_latent       
        }
    
    def compute_total_loss(self, outputs):
        video_normalised = F.normalize(outputs["video_embedding"])
        audio_normalised = F.normalize(outputs["audio_embedding"])

        sim = torch.matmul(video_normalised.squeeze(), audio_normalised.squeeze().T) / self.temperature
        labels = torch.arange(sim.size(0), device=self.device)
        contrastive_loss = F.cross_entropy(sim, labels)
        
        video_attention = outputs["video_attention_map"]
        audio_attention = outputs["audio_attention_map"]

        # IDK how to compute this - basically we want to measure "how much" the modalities' features attend to another (well aligned reprs will attend more (non-diagonal non-normal))
        # temporal_loss = 
        temporal_loss = 0


        return {
            "loss": contrastive_loss + temporal_loss,
            "contrastive_loss": contrastive_loss,
            "temporal_loss": temporal_loss
        }

def train_step(model, features, optimizer):
    outputs = model(features)
    loss = model.compute_total_loss(outputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "total_loss": loss["loss"],
        "contrastive_loss": loss["contrastive_loss"],
        "temporal_loss": loss["temporal_loss"]
    }    

class TemporalCrossAlignmentModule(nn.Module):
    def __init__(self, feature_dim=768, num_heads=8):
        super().__init__()
        self.video_to_audio_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.audio_to_video_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        self.gate = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, video_features, audio_features):
        video_aligned, video_attention = self.video_to_audio_attention(
            query=video_features, key=audio_features, value=audio_features
        )

        audio_aligned, audio_attention = self.audio_to_video_attention(
            query=audio_features, key=video_features, value=video_features
        )

        # add em together using a gate so we can decide how much of each to use rather than just concat
        fused_video = torch.tanh(self.gate(torch.cat([video_features, video_aligned], dim=-1)))
        fused_audio = torch.tanh(self.gate(torch.cat([audio_features, audio_aligned], dim=-1)))

        return fused_video, fused_audio, video_attention, audio_attention


if __name__ == "__main__":
    from feature_extraction import MultiModalFeatureExtractor
    import os
    
    extractor = MultiModalFeatureExtractor()
    model = SharedLatentSpace()
    optimizer = torch.optim.Adam(model.parameters())
    
    audio_directory = "../Solos/processed_audios/Cello"
    video_directory = "../Solos/processed_videos/Cello"
    audio_path = os.path.join(audio_directory, "-qRn8UyHogA_segment_1.mp3")
    video_path = os.path.join(video_directory, "-qRn8UyHogA.f136_segment_1.mp4")
    
    features = extractor.extract_features(video_path, audio_path)
    
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            features[k] = v.to(model.device)

    with torch.no_grad():
        outputs = model.encode(features)
        print("Output Shapes:", {k: v.shape for k, v in outputs.items()})
        loss = model.compute_total_loss(outputs)
        print("Untrained total loss: ", loss["loss"])
        print("Untrained contrastive loss: ", loss["contrastive_loss"])
        print("Untrained temporal loss: ", loss["temporal_loss"])
    
    # loss = train_step(model, features, optimizer)
    # print("Training loss:", loss)