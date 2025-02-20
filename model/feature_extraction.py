import torch, os
import torch.nn as nn
from audio_feature_extraction import TemporalAudioEncoder
from video_feature_extraction import ViViTFeatureExtractor


class MultiModalFeatureExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.video_extractor = ViViTFeatureExtractor(device=self.device)
        self.audio_extractor = TemporalAudioEncoder(target_length=1568)
        
        self.to(self.device)

        
    def extract_features(self,video_path,audio_path, extract_optical_flow=True):
        video_outputs = self.video_extractor.extract_features(video_path)
        video_features = video_outputs["video_embeddings"].to(self.device)
        optical_flow = video_outputs["optical_flow"].to(self.device) if extract_optical_flow else None
        
        audio_outputs = self.audio_extractor.extract_features(audio_path)
        audio_spectograms = audio_outputs["mel_features"].to(self.device)
        encodec_features = audio_outputs["encodec_features"].to(self.device)
        audio_scales = audio_outputs["audio_scales"].to(self.device)
        
        if video_features.dim() == 2:
            video_features = video_features.unsqueeze(0)
            
        return {
            "video_features": video_features,
            "audio_spectograms": audio_spectograms,
            "optical_flow": optical_flow,
            "encodec_features": encodec_features,
            "audio_scale": audio_scales,
            "additional_info": {
                "padding_mask": audio_outputs["padding_mask"],
                "original_length": audio_outputs["original_length"]
            }
        }
    
if __name__ == "__main__":
    # test the multimodal feature extractor
    audio_directory = "../Solos/processed_audios/Cello"
    video_directory = "../Solos/processed_videos/Cello"

    audio_path = os.path.join(audio_directory, "-qRn8UyHogA_segment_1.mp3")
    video_path = os.path.join(video_directory, "-qRn8UyHogA.f136_segment_1.mp4")

    extractor = MultiModalFeatureExtractor()
    features = extractor.extract_features(video_path, audio_path)

    print("Video Features Shape: ", features["video_features"].shape)
    print("Audio Features Shape: ", features["audio_features"].shape)
    print("Optical Flow Shape: ", features["optical_flow"].shape)
    print("Encodec Features Shape: ", features["encodec_features"].shape)
    print("Original Mel Shape: ", features["additional_info"]["original_mel"].shape)
    print("Original Length: ", features["additional_info"]["original_length"])