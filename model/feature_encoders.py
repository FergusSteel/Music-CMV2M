import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EncodecTrainer:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # Change MSE to CrossEntropy for classification
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=1e-4
        )

    def train_step(self, batch):
        self.optimizer.zero_grad()

        # Forward pass
        latent = self.encoder(batch)
        reconstruction = self.decoder(latent)  # [batch, 4, 500, codebook_size]

        # Reshape input and output for CrossEntropyLoss
        target = batch.view(batch.size(0), 4, 500)
        if reconstruction.dim() == 5:  # [batch, 4, 500, 32, codebook_size]
            # If we have 5 dimensions, adjust permute accordingly
            target = batch.view(batch.size(0), 4, 500)
            reconstruction = reconstruction.view(batch.size(0), 4, 500, -1)
            reconstruction = reconstruction.permute(0, 3, 1, 2)
        else:  # [batch, 4, 500, codebook_size]
            target = batch.view(batch.size(0), 4, 500)
            reconstruction = reconstruction.permute(0, 3, 1, 2)

        loss = self.criterion(reconstruction, target)

        loss.backward()
        self.optimizer.step()

        return loss.item()

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
        
        self.conv = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)
        
        self.to(self.device)
    
    def forward(self, x):
        x = x.squeeze(2)
        batch_size = x.size(0)
        seq_length = x.size(1)
        encoded_sequences = []
        
        for i in range(batch_size):
            single_sequence = x[i]  # [15, 2, 224, 224]
            
            encoded_frames = []
            for j in range(seq_length):
                frame = single_sequence[j]  # [2, 224, 224]
                
                conv_out = self.conv(frame)  # [1, 64, 112, 112]
                pooled = self.pool(conv_out)  # [1, 64, 7, 7]
                flattened = pooled.view(1, -1)  # [1, 64 * 7 * 7]
                encoded = self.fc(flattened)  # [1, latent_dim]
                
                encoded_frames.append(encoded)

            sequence_encoding = torch.cat(encoded_frames, dim=0)
            encoded_sequences.append(sequence_encoding)
        
        final_output = torch.stack(encoded_sequences, dim=0)
        return final_output


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
        # Input shape: [batch_size, 15, latent_dim]
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Reshape to process all sequences at once
        x = x.view(-1, x.size(-1))  # [batch_size * 15, latent_dim]
        
        x = self.fc(x)  # [batch_size * 15, 64 * 7 * 7]
        x = x.view(-1, 64, 7, 7)  # [batch_size * 15, 64, 7, 7]
        x = self.deconv(x)  # [batch_size * 15, 2, H, W]
        x = self.upsample(x)  # [batch_size * 15, 2, 224, 224]
        x = x.view(batch_size, 15, 2, 224, 224)  # [batch_size, 15, 2, 224, 224]
        x = x.unsqueeze(2)

        
        return x
    
# class EncodecEncoder(nn.Module):
#     def __init__(self, latent_dim=768):
#         super().__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         # simple projection
#         self.fc = nn.Linear(4 * 500, latent_dim)
        
#         self.to(self.device)
    
#     def forward(self, x):
        
#         x = x.type(torch.FloatTensor).to(self.device)
        
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)  # [batch_size, 4*500]
        
#         x = self.fc(x)  # [batch_size, latent_dim]
        
#         # Add back sequence dimension to match other encoders
#         x = x.unsqueeze(1)  # [batch_size, 1, latent_dim]
#         print("encoded encodec shape", x.shape)
#         return x

# class EncodecDecoder(nn.Module):
#     def __init__(self, latent_dim=768):
#         super().__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         self.fc = nn.Linear(latent_dim, 4 * 500)
        
#         self.to(self.device)
    
#     def forward(self, x):
#         # Input shape: [batch_size, 1, latent_dim]

#         x = x.type(torch.FloatTensor).to(self.device)
        
#         batch_size = x.size(0)
#         x = x.squeeze(1)  # [batch_size, latent_dim]
        
#         x = self.fc(x)  # [batch_size, 4*500]
        
#         # Reshape to match input format with all dimensions
#         x = x.view(batch_size, 1, 1, 4, 500)  # [batch_size, 1, 1, 4, 500]
#         print("Decoder final shape:", x.shape)
#         return x
class EncodecEncoder(nn.Module):
    def __init__(self, latent_dim=768, codebook_size=2048):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(codebook_size, 32)
        self.fc1 = nn.Linear(500*32, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, latent_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.to(self.device)

    def forward(self, x):
        x = x.long().to(self.device)
        batch_size = x.size(0)

        x = x.view(batch_size, 4, 500)

        x_embedded = []
        for i in range(4):
            embedded = self.embedding(x[:, i])
            x_embedded.append(embedded)

        x_embedded = torch.stack(x_embedded, dim=1)
        x = x_embedded.view(batch_size, 4, 500*32)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        print("encoded shape", x.shape)
        return x

class EncodecDecoder(nn.Module):
    def __init__(self, latent_dim=768, codebook_size=2048):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.codebook_size = codebook_size

        self.fc1 = nn.Linear(latent_dim, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 500*32)
        self.final_proj = nn.Linear(32, codebook_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.final_proj.weight)

        self.to(self.device)

    def forward(self, x):
        x = x.float().to(self.device)
        batch_size = x.size(0)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.norm1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = x.view(batch_size, 4, 500, 32)

        logits = self.final_proj(x)

        if self.training:
            return logits
        else:
            return torch.argmax(logits, dim=-1).view(batch_size, 1, 1, 4, 500)
    
class SpectrogramEncoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Convolutional backbone for processing 2D spectrogram data
        # Input shape: [batch_size, 1, 128, 2001] (channels, mel bins, time frames)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # [batch, 32, 64, 1000]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # [batch, 64, 32, 500]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # [batch, 128, 16, 250]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # [batch, 256, 8, 125]
        
        # Normalization and activation
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.activation = nn.LeakyReLU(0.2)
        
        # Global pooling to reduce spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # [batch, 256, 4, 4]
        
        # Final projection to latent
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
        
        self.to(self.device)
    
    def forward(self, x):
        # Input shape: [batch_size, 128, 2001]
        x = x.float().to(self.device)
        
        # Add channel dimension if not present
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # [batch, 1, 128, 2001]
            
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.activation(x)
        
        # Adaptive pooling to fixed dimensions
        x = self.adaptive_pool(x)  # [batch, 256, 4, 4]
        
        # Flatten and project to latent
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # [batch, 256*4*4]
        x = self.fc(x)  # [batch, latent_dim]

        return x

class SpectrogramDecoder(nn.Module):
    def __init__(self, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Projection from latent to initial feature map
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Transposed convolutions for upsampling
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # [batch, 128, 8, 8]
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # [batch, 64, 16, 16]
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # [batch, 32, 32, 32]
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)    # [batch, 16, 64, 64]
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)     # [batch, 1, 128, 128]
        
        # Normalization and activation
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(32)
        self.norm4 = nn.BatchNorm2d(16)
        self.activation = nn.LeakyReLU(0.2)
        
        # Final activation (tanh to get values in [-1, 1] range which can be rescaled)
        self.final_activation = nn.Tanh()
        
        # Final upsampling to target dimensions
        self.final_upsample = nn.Upsample(size=(128, 2001), mode='bilinear', align_corners=False)
        
        self.to(self.device)
    
    def forward(self, x):
        # Input shape: [batch_size, latent_dim]
        x = x.float().to(self.device)
        
        batch_size = x.size(0)
        
        # Project to initial feature map
        x = self.fc(x)
        x = x.view(batch_size, 256, 4, 4)  # [batch, 256, 4, 4]
        
        # Apply transposed convolutions
        x = self.deconv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        x = self.deconv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        
        x = self.deconv4(x)
        x = self.norm4(x)
        x = self.activation(x)
        
        x = self.deconv5(x)
        x = self.final_activation(x)
        
        # Final upsampling to target dimensions
        x = self.final_upsample(x)  # [batch, 1, 128, 2001]
        
        # Remove channel dimension to match input
        x = x.squeeze(1)  # [batch, 128, 2001]

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
        
        reconstruction = reconstruction.float()
        batch = batch.float()
        loss = self.criterion(reconstruction, batch)

        print(type(loss.item()))
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
    

def train_encoder(modality, instrument_type, encoder, decoder, num_epochs=100, batch_size=4):
    print(f"\n=== {modality} Encoder Training ===")

    # Create data loaders
    train_loader, val_loader = create_training_loaders(
        feature_dir=f"..\\{instrument_type}_features\\{modality}",
        modality=modality,
        batch_size=batch_size
    )

    # Get a sample batch to check shapes
    for batch in train_loader:
        print(f"Sample batch shape: {batch.shape}")
        break

    # Initialize trainer and tracking variables
    if modality == "encodec":
        trainer = EncodecTrainer(encoder, decoder, encoder.device)
    else:
        trainer = EncoderTrainer(encoder, decoder)
    best_val_loss = float('inf')

    # Main training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            loss = trainer.train_step(batch)
            epoch_loss += loss
            num_batches += 1

            # Optional: print batch-level progress
            if (num_batches % 10 == 0) or (num_batches == 1):
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {num_batches} Loss: {loss:.4f}")

        # Calculate average losses
        avg_train_loss = epoch_loss / num_batches
        if modality != "encodec":
            val_loss = trainer.validate(val_loader)
        else:
            val_loss = avg_train_loss

        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best model! Validation loss: {val_loss:.4f}")
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'val_loss': val_loss
            }, f'best_model_{instrument_type}_{modality}_{type(encoder).__name__}.pt')

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return best_val_loss

from feature_extraction import MultiModalFeatureExtractor
from dataloader import (FeatureExtractionDataset, extract_and_save_features, create_training_loaders)


def extract_features(instrument_type):
    print("\n=== Feature Extraction ===")

    # Initialize feature extractor
    extractor = MultiModalFeatureExtractor()

    # Extract and cache features
    extract_and_save_features(
        data_dir="..\\dat",
        instrument_type=instrument_type,
        feature_extractor=extractor,
        output_dir=f"..\\{instrument_type}_features"
    )

if __name__ == "__main__":
    dat_directory = "..\\dat"

    instrument = "Violin"
    extract_features(instrument)

    for modality in ["optical_flow", "encodec", "spectrogram"]:
        if modality == "optical_flow":
            continue
            encoder = OpticalFlowEncoder()
            decoder = OpticalFlowDecoder()
        if modality == "encodec":
            continue
            encoder = EncodecEncoder()
            decoder = EncodecDecoder()
        if modality == "spectrogram":
            encoder = SpectrogramEncoder()
            decoder = SpectrogramDecoder()

        train_encoder(modality, instrument, encoder, decoder, num_epochs=1, batch_size=8)
