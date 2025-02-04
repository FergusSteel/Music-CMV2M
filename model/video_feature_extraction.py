import torch
import torchvision.transforms as transforms
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

    def extract_features(self, video_path):
        frames = self._load_video_frames(video_path)
        print("Frame Shape: ", frames.shape)
        video_embeddings = self._extract_vivit_embeddings(frames)
        optical_flow = self._compute_optical_flow(frames)

        combined_features = []

        return {
            "video_embeddings": video_embeddings,
            "optical_flow": optical_flow
        }

    def _load_video_frames(self, video_path):
        video, _, _ = read_video(video_path, pts_unit="sec")
        video_frames = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        indices = torch.linspace(0, video_frames.shape[0] - 1, 16).long()
        video_frames = video_frames[indices]
        return video_frames

    def _extract_vivit_embeddings(self, frames):
        with torch.no_grad():
            inputs = self.image_processor(list(frames), return_tensors="pt")
            outputs = self.model(**inputs).last_hidden_state
        return outputs

    def _compute_optical_flow(self, frames):
        video_frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C ] - rehaspeeeeee
        flows = []

        for i in range(video_frames.shape[0] - 1):
            prev_frame = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY)
            next_frame = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flows.append(flow)

        return torch.tensor(flows)

    def _combine_features(self, video_embeddings, optical_flow):
        flow_features = torch.tensor(optical_flow).mean(dim=(-1, -2)) 
        flow_features = flow_features.unsqueeze(-1).repeat(1, video_embeddings.shape[-1]) 
        combined = video_embeddings[:-1] + flow_features.to(video_embeddings.device)
        return combined

if __name__ == "__main__":
    video_path = "../../Solos/processed_videos/Cello/-qRn8UyHogA.f136_segment_1.mp4"
    extractor = ViViTFeatureExtractor()
    features = extractor.extract_features(video_path)
    print("Video Embeddings Shape:", features["video_embeddings"].size())
    print("Optical Flow Shape:", features["optical_flow"].size())
    print("Combined Features Shape:", features["combined_features"])
