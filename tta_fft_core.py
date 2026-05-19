import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTChannelGate(nn.Module):
    def __init__(self, channels, radius=8):
        super().__init__()
        self.channels = channels
        self.radius = radius
        
        # MLP for channel weights: 2*C -> C//reduction -> C
        reduction = 4
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
        # Initialize last layer to bias=4.6 to make Sigmoid output ~0.99
        # This ensures Identity Mapping on the first forward pass
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.constant_(self.mlp[2].bias, 4.6)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Compute 2D FFT
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"))
        
        # 2. Compute magnitude
        magnitude = torch.log1p(torch.abs(spectrum))
        
        # 3. Create low/high frequency masks
        device = x.device
        center_y, center_x = H // 2, W // 2
        Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist = torch.sqrt((Y - center_y)**2 + (X - center_x)**2)
        
        low_mask = (dist <= self.radius).float().view(1, 1, H, W)
        high_mask = 1.0 - low_mask
        
        # 4. Compute average energy descriptors
        low_count = low_mask.sum()
        high_count = high_mask.sum()
        
        low_descriptor = (magnitude * low_mask).sum(dim=(2, 3)) / (low_count + 1e-8)  # [B, C]
        high_descriptor = (magnitude * high_mask).sum(dim=(2, 3)) / (high_count + 1e-8) # [B, C]
        
        # 5. Concat
        desc = torch.cat([low_descriptor, high_descriptor], dim=1) # [B, 2C]
        
        # 6. Pass through MLP
        weights = self.mlp(desc) # [B, C]
        weights = torch.sigmoid(weights).view(B, C, 1, 1)
        
        return weights

def alignment_loss(gated_skip1, vit_output):
    """
    gated_skip1: [B, 256, 56, 56] - Noisy skip connection after gate
    vit_output: [B, 512, 14, 14]  - Clean semantic features from ViT
    """
    # 1. Downsample gated_skip1 to match ViT spatial resolution
    gated_down = F.adaptive_avg_pool2d(gated_skip1, vit_output.shape[-2:]) # [B, 256, 14, 14]
    
    # 2. Extract Spatial Attention Maps (SAM) by averaging across channels
    skip_sam = gated_down.mean(dim=1, keepdim=True) # [B, 1, 14, 14]
    vit_sam = vit_output.mean(dim=1, keepdim=True) # [B, 1, 14, 14]
    
    # 3. Flatten
    B = skip_sam.shape[0]
    skip_flat = skip_sam.view(B, -1)
    vit_flat = vit_sam.view(B, -1)
    
    # 4. Cosine Similarity Loss
    cos_sim = F.cosine_similarity(skip_flat, vit_flat, dim=1) # [B]
    loss = 1.0 - cos_sim.mean()
    
    return loss
