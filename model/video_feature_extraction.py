import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import optical_flow
from torchvision.io import read_video
import cv2
import av
import numpy as np
from transformers import VivitImageProcessor, VivitModel, AutoImageProcessor, VideoMAEModel
from huggingface_hub import hf_hub_download

class ViViTFeatureExtractor:
    def __init__(self, vivit_model_name="google/vivit-b-16x2-kinetics400", device=None, vivit=False):
        if vivit:
            self.model = VivitModel.from_pretrained(vivit_model_name)
            self.image_processor = VivitImageProcessor.from_pretrained(vivit_model_name)
        else:
            self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            print(self.model.config.num_frames)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.flow_model = optical_flow.raft_large().to(self.device)
        self.flow_model.eval()

    def extract_features(self, video_path):
        frames = self.load_frames(video_path)
        print("Frame Shape: ", frames.shape)
        video_embeddings = self.extract_embeddings(frames)
        optical_flow = self.compute_optical_flow(frames)

        return {
            "video_embeddings": video_embeddings,
            "optical_flow": optical_flow
        }

    def load_frames(self, video_path):
        video, _, _ = read_video(video_path, pts_unit="sec")
        video_frames = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        indices = torch.linspace(0, video_frames.shape[0] - 1, 16).long()
        video_frames = video_frames[indices]
        return video_frames

    def extract_embeddings(self, frames):
        with torch.no_grad():
            inputs = self.image_processor(list(frames), return_tensors="pt")
            outputs = self.model(**inputs).last_hidden_state
        return outputs

    def compute_optical_flow(self, frames):
        flows = []
        frames = frames.to(self.device)
        
        with torch.no_grad():
            for i in range(frames.shape[0] - 1):
                frame1 = frames[i:i+1] * 255.0
                frame2 = frames[i+1:i+2] * 255.0
                flow_predictions = self.flow_model(frame1, frame2)
                final_flow = flow_predictions[-1] # discard the other predictions only use the last one from da raft model
                flows.append(final_flow)

        return torch.stack(flows)

if __name__ == "__main__":
    video_path = "../Solos/processed_videos/Cello/-qRn8UyHogA.f136_segment_1.mp4"
    extractor = ViViTFeatureExtractor()
    features = extractor.extract_features(video_path)
    print("Video Embeddings Shape:", features["video_embeddings"].size())
    print("Optical Flow Shape:", features["optical_flow"].size())
