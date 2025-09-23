import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from elucidated_diffusion.elucidated_diffusion import sigma_data, sigma_max, sigma_min, P_mean, P_std

class FrequencyDomainLoss(nn.Module):
    """
    Frequency Domain Perceptual Loss (FDPL) adapted for EDM framework.
    
    Based on:
    - Sims et al. "Frequency Domain-based Perceptual Loss for Super Resolution" (2020)
    - Fuoli et al. "Fourier Space Losses for Efficient Perceptual Image Super-Resolution" (2021)

    https://claude.ai/public/artifacts/d5cc312c-33f0-44e2-80f0-25c57e03cd4b
    https://claude.ai/share/deb80cba-8f15-416d-b1b1-e714e4f91df2
    """
    def __init__(self, weight_low_freq=1.0, weight_high_freq=2.0, alpha=1.0):
        super().__init__()
        self.weight_low_freq = weight_low_freq
        self.weight_high_freq = weight_high_freq  
        self.alpha = alpha  # Controls frequency emphasis
        
    def create_frequency_mask(self, shape, device):
        """Create frequency weighting mask that emphasizes perceptually important frequencies"""
        H, W = shape[-2:]
        
        # Create coordinate grids
        u = torch.arange(H, device=device, dtype=torch.float32) - H // 2
        v = torch.arange(W, device=device, dtype=torch.float32) - W // 2
        U, V = torch.meshgrid(u, v, indexing='ij')
        
        # Radial frequency (distance from DC component)
        freq_radius = torch.sqrt(U**2 + V**2)
        
        # Normalize to [0, 1]
        freq_radius = freq_radius / (torch.sqrt(torch.tensor(H**2 + W**2, device=device)) / 2)
        
        # Create perceptual weighting (emphasize mid frequencies, reduce very high freq noise)
        # This follows human visual system sensitivity
        weight_mask = 1.0 + self.alpha * torch.exp(-(freq_radius - 0.3)**2 / 0.2)
        
        return weight_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    def forward(self, pred, target):
        """
        Compute frequency domain loss between predicted and target images.
        
        Args:
            pred: Predicted image [B, C, H, W] 
            target: Ground truth image [B, C, H, W]
            
        Returns:
            Frequency domain loss (scalar)
        """
        # Convert to frequency domain (per channel)
        pred_freq = torch.fft.fft2(pred, dim=(-2, -1))
        target_freq = torch.fft.fft2(target, dim=(-2, -1))
        
        # Shift zero frequency to center for easier analysis
        pred_freq = torch.fft.fftshift(pred_freq, dim=(-2, -1))
        target_freq = torch.fft.fftshift(target_freq, dim=(-2, -1))
        
        # Compute magnitude spectra (phase is often less perceptually important)
        pred_magnitude = torch.abs(pred_freq)
        target_magnitude = torch.abs(target_freq)
        
        # Create frequency weighting mask
        freq_mask = self.create_frequency_mask(pred.shape, pred.device)
        
        # Apply frequency weighting
        weighted_pred = pred_magnitude * freq_mask
        weighted_target = target_magnitude * freq_mask
        
        # L1 loss in frequency domain (often works better than L2 for perceptual tasks)
        freq_loss = F.l1_loss(weighted_pred, weighted_target)
        
        return freq_loss

class CombinedEDMFrequencyLoss(nn.Module):
    """Combined EDM loss with frequency domain loss"""
    def __init__(self, lambda_freq=0.1, sigma_data=0.5):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.sigma_data = sigma_data
        self.freq_loss = FrequencyDomainLoss()
        
    def edm_loss_weight(self, sigma):
        """EDM loss weighting function"""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    
    def forward(self, pred, target, sigma):
        """
        Combined loss function for EDM + Frequency Domain
        
        Args:
            pred: Model prediction (noise prediction) 
            target: Ground truth clean image
            sigma: Noise level for current batch
        """
        # Standard EDM loss  
        weight = self.edm_loss_weight(sigma.flatten())[:, None, None, None]
        edm_loss = (weight * (pred - target) ** 2).mean()
        
        # Frequency domain loss (applied to final denoised images)
        freq_loss = self.freq_loss(pred, target)
        
        # Combined loss
        total_loss = edm_loss + self.lambda_freq * freq_loss
        
        return total_loss, edm_loss.item(), freq_loss.item()

# Integration with your training loop:
def train_a_batch_with_freq_loss(model_edm, optimizer_edm, batch, combined_loss_fn, device = 'cuda'):
    """Modified training function with frequency domain loss"""
    x = batch
    hr_256, lr_64 = x
    hr_256 = hr_256.to(device)
    lr_64 = lr_64.to(device)

    x = hr_256
    # Sample sigma (your existing code)
    rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    
    # Preconditioning coefficients (your existing code)
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
    c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
    c_noise = sigma.log() / 4
    
    # Add noise
    noise = torch.randn_like(x) * sigma
    y_noisy = x + noise
    
    # Model prediction
    c_noise_input = c_noise.view(x.shape[0])
    F_x = model_edm(c_in * y_noisy, lr_64, c_noise_input)
    
    # Preconditioning: D_x = c_skip * y_noisy + c_out * F_x
    D_x = c_skip * y_noisy + c_out * F_x
    
    # Combined loss (EDM + Frequency)
    total_loss, edm_loss_val, freq_loss_val = combined_loss_fn(D_x, x, sigma)
    
    optimizer_edm.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_edm.parameters(), max_norm=1.0)
    optimizer_edm.step()
    
    return total_loss.item(), edm_loss_val, freq_loss_val

# Usage example:
if __name__ == "__main__":
    # Initialize combined loss
    combined_loss = CombinedEDMFrequencyLoss(lambda_freq=0.1)
    
    # Test the loss function
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256) 
    sigma = torch.randn(2, 1, 1, 1).exp()
    
    total_loss, edm_loss, freq_loss = combined_loss(pred, target, sigma)
    print(f"Total loss: {total_loss:.4f}")
    print(f"EDM loss: {edm_loss:.4f}")
    print(f"Frequency loss: {freq_loss:.4f}")
