import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAlignmentModule(nn.Module):
    def __init__(self, feature_dim=768, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, video_features, audio_features):
        aligned_features, attention_weights = self.cross_attention(
            query=video_features,
            key=audio_features,
            value=audio_features
        )
        return aligned_features, attention_weights

class SharedLatentSpace(nn.Module):
    def __init__(self, feature_dim=768, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.temporal_alignment = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            batch_first=True
        ).to(self.device)

        # Project the videomae features to latent space man
        self.video_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        #Projectin audio to latent space 
        self.audio_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Encodec token encoder
        self.token_encoder = nn.Sequential(
            nn.Linear(2000, latent_dim),  # 4*500 flattened tokens
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Temperature parameter for contrastive loss
        self.token_embedding = nn.Embedding(2048, latent_dim).to(self.device)
        self.token_proj = nn.Linear(latent_dim, latent_dim).to(self.device)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        self.to(self.device)
    
    def encode_video(self, video_features):
        return self.video_encoder(video_features)
    
    def encode_audio(self, audio_features, encodec_features):
        # Encode aligned features
        mel_latent = self.audio_encoder(audio_features)
        B, _, C, L = encodec_features.shape
        token_latent = self.token_embedding(encodec_features.view(-1)).view(B, 1, C, L, -1)
        token_latent = token_latent.mean(dim=(1,2,3))  # Average across time and channels
        token_latent = self.token_proj(token_latent)
        
        audio_latent = mel_latent + token_latent.unsqueeze(1).expand_as(mel_latent)
        
        return audio_latent
    
    def align_modalities(self, video_latent, audio_latent):
        aligned_features, attention_map = self.temporal_alignment(
            query=video_latent,
            key=audio_latent,  
            value=audio_latent
        )
    
        
        return aligned_features, audio_latent, attention_map
        
    def forward(self, features):
        # Encode both modalities
        video_latent = self.encode_video(features["video_features"])
        audio_latent = self.encode_audio(
            features["audio_features"],
            features["encodec_features"]
        )
        video_embed, audio_embed, attention = self.align_modalities(video_latent, audio_latent)
        
        return {
            "video_embedding": video_embed,     
            "audio_embedding": audio_embed,      
            "attention_map": attention,          
            "video_latent": video_latent,       
            "audio_latent": audio_latent       
        }
    
    def compute_total_loss(self, outputs):
        video_normalised = F.normalize(outputs["video_latent"])
        audio_normalised = F.normalize(outputs["audio_latent"])

        sim = torch.matmul(video_normalised.squeeze(), audio_normalised.squeeze().T) / self.temperature
        labels = torch.arange(sim.size(0), device=self.device)
        contrastive_loss = F.cross_entropy(sim, labels)
        
        attention = outputs["attention_map"]
        temporal_loss = -torch.mean(torch.sum(attention * torch.eye(attention.size(1), device=self.device), dim=1)) # basically we take the diagonal elements of the attention map and if theyre big its bad cos that suggest that they are not temporally aligned
        
        return {
            "loss": contrastive_loss + temporal_loss,
            "contrastive_loss": contrastive_loss,
            "temporal_loss": temporal_loss,
            "attention": attention
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
    s
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            features[k] = v.to(model.device)

    with torch.no_grad():
        outputs = model(features)
        print("Output Shapes:", {k: v.shape for k, v in outputs.items()})
        loss = model.compute_total_loss(outputs)
        print("Untrained total loss: ", loss["loss"])
        print("Untrained contrastive loss: ", loss["contrastive_loss"])
        print("Untrained temporal loss: ", loss["temporal_loss"])
    
    # loss = train_step(model, features, optimizer)
    # print("Training loss:", loss)