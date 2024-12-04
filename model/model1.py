import torch
import torch.nn as nn
from transformers import ViTModel

# Fergus Steel (2542391s) MSCi Project - This is a basic prototype model to demonstrate the concept of cross-modal audio-visual translation using a simple feedforward network

class VisualAudioModel(nn.Module):
    def __init__(self):
        super(VisualAudioModel, self).__init__()

        # Load a pretrained Visual Transformer from Hugging Face
        self.visual_model = ViTModel.from_pretrained('google/vit-base-patch16-224')

        self.audio_network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU()
        )
        self.fc = nn.Linear(768 + 512, 1024)
        self.output_layer = nn.Linear(1024, 128 * 7 * 7)

    def forward(self, video_data, audio_data):
        # Get data embeddings
        visual_embeddings = self.visual_model(pixel_values=video_data).last_hidden_state[:, 0, :]
        audio_embeddings = self.audio_network(audio_data)

        # Combine embeddings - this is a simple concatenation, more complex methods can be used (cross attention, we may be using fckin Optical Flow Fields too)
        combined = torch.cat((visual_embeddings, audio_embeddings), dim=1)
        output = self.fc(combined)

        # Generate spectrogram as output
        spectrogram_output = self.output_layer(output)
        spectrogram_output = spectrogram_output.view(-1, 1, 128, 128)

        return spectrogram_output

    def train_model(self, train_loader, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            for video_data, audio_data in train_loader:
                optimizer.zero_grad()

                video_data = video_data.cuda()
                audio_data = audio_data.cuda()

                output = self.forward(video_data, audio_data)
                loss = loss_fn(output, audio_data)  # Compare the generated and ground-truth spectrogram
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

    def evaluate(self, val_loader):
        self.eval()
        with torch.no_grad():
            for video_data, audio_data in val_loader:
                video_data = video_data.cuda()
                audio_data = audio_data.cuda()

                output = self.forward(video_data, audio_data)
                # TODO: Implement evaluation logic

