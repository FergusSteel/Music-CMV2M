import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoFeatureEncoder(nn.module):
    # this will encode video features to the correct latent dimension

    # Training Process:
    # Encode to latent dims, Decode back to original and compare (ensure not too much entropy) - UNET structure Probs
    return

class OpticalFlowEncoder(nn.module):
    # this will encode optical flow to the correct latent dimension
    # So take the 15 frames and fit them into the correct temporal dimension and makes em the right latent dimensions

    # Training Process:
    # Encode to latent dims, Decode back to original and compare (ensure not too much entropy) - UNET structure Probs
    return

class EncodecEncoder(nn.Module):
    # Youll never guess

    # Training Process:
        # Encode to latent dims, Decode back to original and compare AND decode to audio and compare to original audio
    return

class SpectogramEncoder(nn.Module):

    # Training Process:
            # Encode to latent dims, Decode back to original and compare AND SSIM as frequency information is important
    return