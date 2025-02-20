import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VideoFeatureEncoder(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x):
        # already encoded by VideoMAE
        return x

class OpticalFlowEncoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # reduce dimensionality of optical flow
        self.conv = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # project to latent dimension
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.squeeze(1) 
        batch_size = x.size(0)
        
        x = self.conv(x)  # [15, 64, 112, 112]
        x = self.pool(x)  # [15, 64, 7, 7]
        x = x.view(batch_size, -1)  # Flatten: [15, 64 * 7 * 7]
        x = self.fc(x)  # [15, latent_dim]
        
        return x

class EncodecEncoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # simple projection
        self.fc = nn.Linear(4 * 500, latent_dim)
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 
        return x

class SpectrogramEncoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        self.pool = nn.AdaptiveAvgPool2d((128, 128))
        self.fc = nn.Linear(128 * 128, latent_dim)
        
        self.to(self.device)
    
    def forward(self, x):
        # x shape: [1, 128, 2001]
        x = x.unsqueeze(1)  # Add channel dim: [1, 1, 128, 2001]
        x = self.pool(x)  # [1, 1, 128, 128]
        x = x.view(x.size(0), -1)  # Flatten: [1, 128 * 128]
        x = self.fc(x)  # [1, latent_dim] ???
        return x
    
class EncoderTrainer:
    def __init__(self, encoder, decoder, learning_rate=1e-4):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, batch):
        self.encoder.train()
        self.decoder.train()
        
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        latent = self.encoder(batch)
        reconstruction = self.decoder(latent)
        
        loss = self.criterion(reconstruction, batch)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def validate(self, val_loader):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                latent = self.encoder(batch)
                reconstruction = self.decoder(latent)
                loss = self.criterion(reconstruction, batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

# DECODERS
class OpticalFlowDecoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample = nn.Upsample(size=(224, 224))
        
        self.to(self.device)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.deconv(x)
        x = self.upsample(x)
        return x

class EncodecDecoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.fc = nn.Linear(latent_dim, 4 * 500)
        
        self.to(self.device)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 4, 500)
        return x

class SpectrogramDecoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.fc = nn.Linear(latent_dim, 128 * 128)
        self.upsample = nn.Upsample(size=(128, 2001))
        
        self.to(self.device)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 128, 128)
        x = self.upsample(x)
        return x.squeeze(1)

def train_encoder(encoder, decoder, train_loader, val_loader, num_epochs=100):
    trainer = EncoderTrainer(encoder, decoder)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            loss = trainer.train_step(batch)
            epoch_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        val_loss = trainer.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'val_loss': val_loss
            }, f'best_model_{type(encoder).__name__}.pt')

if __name__ == "__main__":
    from dataloader import create_data_loaders
    from feature_extraction import MultiModalFeatureExtractor

    dat_directory = "../dat"
    tl, vl = create_data_loaders(dat_directory)

#     features = extractor.extract_features(video_path, audio_path)
#
#         for k, v in features.items():
#             if isinstance(v, torch.Tensor):
#                 features[k] = v.to(model.device)

    for batch in tl:
        print(batch)

#     # For Optical Flow
#     flow_encoder = OpticalFlowEncoder()
#     flow_decoder = OpticalFlowDecoder()
#     # train_encoder(flow_encoder, flow_decoder, flow_train_loader, flow_val_loader)
#
#     # For Encodec
#     encodec_encoder = EncodecEncoder()
#     encodec_decoder = EncodecDecoder()
#     # train_encoder(encodec_encoder, encodec_decoder, encodec_train_loader, encodec_val_loader)
#
#     # For Spectrogram
#     spec_encoder = SpectrogramEncoder()
#     spec_decoder = SpectrogramDecoder()