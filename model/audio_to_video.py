import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torchaudio
from tqdm import tqdm
from transformers import EncodecModel, AutoProcessor

from feature_encoders import VideoFeatureEncoder, OpticalFlowEncoder, EncodecEncoder, SpectrogramEncoder
from latent_space import SharedLatentSpace, TemporalCrossAlignmentModule

class VideoToAudioGenerator(nn.Module):
    """
    Generate Encodec audio codes from video features using the shared latent space.
    This model leverages the cross-modal understanding gained through the SharedLatentSpace.
    """
    def __init__(self, feature_dim=768, latent_dim=768):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Use the shared latent space for feature encoding and alignment
        self.shared_latent = SharedLatentSpace(
            feature_dim=feature_dim,
            latent_dim=latent_dim
        )

        # Audio decoder specific layers
        self.audio_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU()
        )

        # Predict encodec audio codes
        # Encodec typically has 4 codebooks of size 2048 or similar
        self.code_predictors = nn.ModuleList([
            nn.Linear(latent_dim, 2048) for _ in range(4)
        ])

        self.to(self.device)

    def forward(self, features, training=False):
        """
        Generate audio codes from video features.

        Args:
            features: Dictionary containing video and optionally audio features
            training: Whether in training mode (uses both modalities) or inference mode (video only)

        Returns:
            Dictionary with generated audio codes and additional outputs
        """
        # If in training mode and audio features are available, use alignment
        if training and "audio_spectograms" in features and "encodec_features" in features:
            # Use both modalities and alignment for better learning
            latent_outputs = self.shared_latent.encode(features, align_modalities=True)
            video_embedding = latent_outputs["video_embedding"]
        else:
            # In inference mode, we only have video features
            # Create a features dict with just video components
            video_features = {
                "video_features": features["video_features"],
                "optical_flow": features["optical_flow"]
            }

            # Add dummy audio tensors if they don't exist (for the shared latent space encoder)
            if "audio_spectograms" not in video_features:
                batch_size = features["video_features"].shape[0]
                video_features["audio_spectograms"] = torch.zeros(
                    (batch_size, 128, 128), device=self.device
                )
                video_features["encodec_features"] = torch.zeros(
                    (batch_size, 1, 1, 4, 500), device=self.device
                )

            # Encode just the video features without alignment
            latent_outputs = self.shared_latent.encode(video_features, align_modalities=False)
            video_embedding = latent_outputs["video_embedding"]

        # Decode video embedding to audio
        audio_features = self.audio_decoder(video_embedding)

        # Predict codes for each codebook
        batch_size = audio_features.shape[0]
        seq_len = audio_features.shape[1]

        # Generate logits for each codebook
        code_logits = []
        for predictor in self.code_predictors:
            logits = predictor(audio_features)  # [batch, seq, 2048]
            code_logits.append(logits)

        # Stack along new dimension to get [batch, seq, num_codebooks, num_classes]
        stacked_logits = torch.stack(code_logits, dim=2)

        # Get the predicted codes (argmax)
        predicted_codes = torch.argmax(stacked_logits, dim=-1)  # [batch, seq, num_codebooks]

        # Create scale factors (usually all ones)
        audio_scales = torch.ones((batch_size, seq_len), device=self.device)

        return {
            "audio_codes": predicted_codes,
            "audio_logits": stacked_logits,
            "audio_scales": audio_scales,
            "latent_outputs": latent_outputs
        }

    def compute_loss(self, outputs, target_codes):
        """
        Compute the losses for training.

        Args:
            outputs: Dictionary from forward pass
            target_codes: Ground truth encodec codes [batch, frames, codebooks]

        Returns:
            Dictionary of losses
        """
        # Cross entropy loss for code prediction
        logits = outputs["audio_logits"]  # [batch, frames, codebooks, classes]
        batch_size, seq_len, num_codebooks, num_classes = logits.shape

        # Reshape for cross entropy loss
        logits_flat = logits.reshape(-1, num_classes)
        target_flat = target_codes.reshape(-1).long()

        # Compute cross entropy loss
        ce_loss = F.cross_entropy(logits_flat, target_flat)

        # If we have latent outputs, leverage contrastive loss too
        total_loss = ce_loss
        latent_loss = None

        if "latent_outputs" in outputs and outputs["latent_outputs"]["video_attention_map"] is not None:
            latent_loss = self.shared_latent.compute_total_loss(outputs["latent_outputs"])

            # Add latent space losses with weighting
            total_loss = ce_loss + 0.5 * latent_loss["loss"]

        losses = {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
        }

        if latent_loss is not None:
            losses.update({
                "latent_loss": latent_loss["loss"],
                "contrastive_loss": latent_loss["contrastive_loss"],
                "temporal_loss": latent_loss["temporal_loss"]
            })

        return losses


class VideoToAudioExtractor:
    """
    End-to-end pipeline for extracting audio from videos.
    Combines feature extraction, generator model, and audio decoding.
    """
    def __init__(self,
                 generator=None,
                 encodec_model_name="facebook/encodec_32khz",
                 generator_checkpoint=None,
                 device=None,
                 sample_rate=32000):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        print(f"Using device: {self.device}")

        # Initialize generator model
        self.generator = generator or VideoToAudioGenerator().to(self.device)

        # Load generator checkpoint if provided
        if generator_checkpoint and os.path.exists(generator_checkpoint):
            checkpoint = torch.load(generator_checkpoint, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded generator checkpoint from {generator_checkpoint}")

        # Initialize encodec model for decoding
        self.encodec_model = EncodecModel.from_pretrained(encodec_model_name).to(self.device)
        self.audio_processor = AutoProcessor.from_pretrained(encodec_model_name)

        # Set models to evaluation mode
        self.generator.eval()
        self.encodec_model.eval()

    def extract_audio_from_video(self, video_features, output_path=None):
        """
        Extract audio from video features.

        Args:
            video_features: Dictionary containing video features and optical flow
            output_path: Path to save the generated audio (optional)

        Returns:
            Generated audio waveform
        """
        # Ensure features are on the correct device
        for key, tensor in video_features.items():
            if isinstance(tensor, torch.Tensor):
                video_features[key] = tensor.to(self.device)

        # Generate audio codes
        print("Generating audio codes...")
        with torch.no_grad():
            outputs = self.generator(video_features, training=False)

            # Decode the generated codes to audio
            print("Decoding to audio waveform...")
            audio_values = self.encodec_model.decode(
                audio_codes=outputs['audio_codes'],
                audio_scales=outputs['audio_scales']
            ).audio_values

        # Save audio if output path is provided
        if output_path:
            audio_np = audio_values.cpu().squeeze().numpy()
            torchaudio.save(
                output_path,
                torch.tensor(audio_np).unsqueeze(0),
                self.sample_rate
            )
            print(f"Audio saved to {output_path}")

        return audio_values

    def train(self,
              train_loader,
              val_loader=None,
              num_epochs=10,
              learning_rate=1e-4,
              checkpoint_dir="checkpoints",
              log_interval=10):
        """
        Train the generator model on paired video-audio data.

        Args:
            train_loader: DataLoader with video features and target audio
            val_loader: Optional validation DataLoader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
            log_interval: How often to log training progress

        Returns:
            Dictionary with training history
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set to training mode
        self.generator.train()

        # Initialize optimizer
        optimizer = optim.AdamW(self.generator.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_val_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            train_losses = []

            # Training
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                for key, tensor in batch.items():
                    if isinstance(tensor, torch.Tensor):
                        batch[key] = tensor.to(self.device)

                # Forward pass
                outputs = self.generator(batch, training=True)

                # Compute loss
                loss_dict = self.generator.compute_loss(outputs, batch["encodec_features"])
                loss = loss_dict['total_loss']

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track loss
                train_losses.append(loss.item())

                # Update progress bar
                if batch_idx % log_interval == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Calculate average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)

                # Update learning rate
                scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': self.generator.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

            # Log progress
            if val_loader:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        # Set back to eval mode
        self.generator.eval()

        return history

    def validate(self, val_loader):
        """
        Validate the model on a validation set.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Average validation loss
        """
        self.generator.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                for key, tensor in batch.items():
                    if isinstance(tensor, torch.Tensor):
                        batch[key] = tensor.to(self.device)

                # Forward pass
                outputs = self.generator(batch, training=True)

                # Compute loss
                loss_dict = self.generator.compute_loss(outputs, batch["encodec_features"])
                val_losses.append(loss_dict['total_loss'].item())

        # Set back to training mode
        self.generator.train()

        return sum(val_losses) / len(val_losses)

# Example of how to use these classes
if __name__ == "__main__":
    from feature_extraction import MultiModalFeatureExtractor
    import os

    # Initialize components
    extractor = MultiModalFeatureExtractor()
    generator = VideoToAudioGenerator()
    audio_extractor = VideoToAudioExtractor(generator=generator)

    # Example paths
    video_path = "../Solos/processed_videos/Violin/sample_video.mp4"
    output_path = "generated_audio.wav"

    # Extract features
    features = extractor.extract_features(video_path, None)  # No audio path for inference

    # Generate audio
    audio = audio_extractor.extract_audio_from_video(features, output_path)
    print(f"Generated audio shape: {audio.shape}")