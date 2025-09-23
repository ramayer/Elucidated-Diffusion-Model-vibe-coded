import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Reuse your proven components from SR
from elucidated_diffusion.models.claude_sr_ViT import SinusoidalPosEmb, PatchEmbedding, MultiHeadAttention, TransformerBlock

class ViTEnhancedUNet64(nn.Module):
    """
    Enhanced 64x64 diffusion model that combines:
    1. Your proven ViT patch processing (successful in SR)
    2. Your existing UNet structure 
    3. Multi-scale attention like your SR model
    
    Key improvements over UNet128:
    - Multi-head attention instead of single-head
    - ViT processing in bottleneck (like your SR success)
    - Cross-scale feature fusion
    - Patch-based global understanding
    """
    def __init__(self, in_channels=3, base_ch=64, emb_dim=128, 
                 patch_size=4, num_vit_layers=4, num_heads=8,
                 use_cross_scale=True):
        super().__init__()
        self.use_cross_scale = use_cross_scale
        
        # EDM-compatible noise embedding (matches your training loop)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )
        
        # ViT components for global processing (borrowing from your SR success)
        self.patch_embed = PatchEmbedding(
            img_size=64, patch_size=patch_size, 
            in_ch=in_channels, embed_dim=base_ch*4
        )
        
        # ViT blocks for bottleneck processing
        self.vit_blocks = nn.ModuleList([
            TransformerBlock(base_ch*4, num_heads, mlp_ratio=4.0)
            for _ in range(num_vit_layers)
        ])
        
        # Multi-head attention for conv layers (upgrade from your single-head)
        # Ensure channels are divisible by num_heads
        self.mha_32 = MultiHeadSelfAttention(base_ch*2, num_heads=8)  # 128 channels, 8 heads = 16 per head
        self.mha_16 = MultiHeadSelfAttention(base_ch*4, num_heads=8)  # 256 channels, 8 heads = 32 per head
        self.mha_8 = MultiHeadSelfAttention(base_ch*4, num_heads=8)   # 256 channels, 8 heads = 32 per head

        # Traditional UNet path (keeping your proven structure)
        self.inc = ResBlock(in_channels, base_ch, emb_dim)
        self.down1 = ResBlock(base_ch, base_ch*2, emb_dim)
        self.down2 = ResBlock(base_ch*2, base_ch*4, emb_dim) 
        self.down3 = ResBlock(base_ch*4, base_ch*4, emb_dim)
        self.pool = nn.AvgPool2d(2)

        # Cross-scale fusion (inspired by your SR model success)
        if use_cross_scale:
            self.cross_scale_fusion = CrossScaleFusion(
                [base_ch, base_ch*2, base_ch*4, base_ch*4], 
                target_size=16
            )

        # Up path (same structure as your current UNet)
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.up_block3 = ResBlock(base_ch*4 + base_ch*4, base_ch*4, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.up_block2 = ResBlock(base_ch*2 + base_ch*4, base_ch*2, emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.up_block1 = ResBlock(base_ch + base_ch*2, base_ch, emb_dim)
        
        self.up0 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.up_block0 = ResBlock(base_ch + base_ch, base_ch, emb_dim)

        # Output (unchanged)
        self.outc = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        """
        Forward pass compatible with your EDM training loop:
        x: [B, 3, 64, 64] - preconditoned input (c_in * y_noisy)
        t: [B] - noise level embedding (c_noise)
        """
        if t.dim() == 1:
            t = t.float()
        else:
            t = t.squeeze(-1).float()
        t_emb = self.time_mlp(t)

        # ViT processing for global structure (like your SR success)
        global_patches = self.patch_embed(x)  # [B, 256, 256]
        for vit_block in self.vit_blocks:
            global_patches = vit_block(global_patches)
        
        # Reshape back to spatial for integration
        B, N, C = global_patches.shape
        patch_grid = int(N ** 0.5)  # 16 for 64x64 with patch_size=4
        global_features = global_patches.transpose(1, 2).view(
            B, C, patch_grid, patch_grid
        )  # [B, 256, 16, 16]

        # Traditional UNet downsampling path
        x1 = self.inc(x, t_emb)                    # [B, 64, 64, 64]
        x2 = self.down1(self.pool(x1), t_emb)      # [B, 128, 32, 32]
        x3 = self.down2(self.pool(x2), t_emb)      # [B, 256, 16, 16]
        x4 = self.down3(self.pool(x3), t_emb)      # [B, 256, 8, 8]

        # Apply multi-head attention at key resolutions
        x2 = x2 + self.mha_32(x2)  # Enhanced 32x32 features
        x3 = x3 + self.mha_16(x3)  # Enhanced 16x16 features
        x4 = x4 + self.mha_8(x4)   # Enhanced 8x8 features

        # Cross-scale fusion (your SR model's successful pattern)
        if self.use_cross_scale:
            cross_scale_features = self.cross_scale_fusion([x1, x2, x3, x4])
            x3 = x3 + cross_scale_features  # Inject multi-scale info

        # Enhanced bottleneck: combine traditional conv + ViT global understanding
        m = self.pool(x4)  # [B, 256, 4, 4]
        
        # Integrate global ViT features with local conv features
        global_upsampled = F.interpolate(
            global_features, size=(4, 4), mode='bilinear', align_corners=False
        )
        m = m + global_upsampled  # Combine local + global understanding

        # Traditional UNet upsampling (unchanged from your working version)
        u3 = self.up_block3(torch.cat([self.up3(m), x4], dim=1), t_emb)
        u2 = self.up_block2(torch.cat([self.up2(u3), x3], dim=1), t_emb) 
        u1 = self.up_block1(torch.cat([self.up1(u2), x2], dim=1), t_emb)
        u0 = self.up_block0(torch.cat([self.up0(u1), x1], dim=1), t_emb)

        return self.outc(u0)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head attention for conv features (upgrade from your single-head)"""
    def __init__(self, channels, num_heads=8, dropout=0.0):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H*W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        h = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(h)


class CrossScaleFusion(nn.Module):
    """Cross-scale fusion from your successful SR model"""
    def __init__(self, channels_list, target_size):
        super().__init__()
        self.target_size = target_size
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, channels_list[0], 1) for ch in channels_list
        ])
        self.fusion = nn.Conv2d(len(channels_list) * channels_list[0], channels_list[0], 1)

    def forward(self, feature_list):
        aligned_features = []
        for i, (feat, proj) in enumerate(zip(feature_list, self.projections)):
            feat = proj(feat)
            if feat.shape[-1] != self.target_size:
                feat = F.interpolate(feat, size=(self.target_size, self.target_size), 
                                   mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        fused = torch.cat(aligned_features, dim=1)
        return self.fusion(fused)


class ResBlock(nn.Module):
    """Your existing ResBlock - keeping it unchanged for compatibility"""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.relu(self.conv2(h))
        return h + self.skip(x)


# Factory function for easy experimentation
def create_vit_enhanced_unet(config='balanced'):
    """
    Create different variants:
    - lightweight: Fewer ViT layers, smaller patches
    - balanced: Good performance/speed tradeoff  
    - heavy: Maximum ViT processing
    """
    configs = {
        'lightweight': {
            'base_ch': 48,
            'patch_size': 8,  # Fewer patches
            'num_vit_layers': 2,
            'num_heads': 8,   # 48*4 = 192, 192÷8 = 24 per head
            'use_cross_scale': False
        },
        'balanced': {
            'base_ch': 64,
            'patch_size': 4,  # Same as your successful SR
            'num_vit_layers': 4,
            'num_heads': 8,   # 64*4 = 256, 256÷8 = 32 per head
            'use_cross_scale': True
        },
        'heavy': {
            'base_ch': 96,
            'patch_size': 2,  # Very fine patches
            'num_vit_layers': 6,
            'num_heads': 12,  # 96*4 = 384, 384÷12 = 32 per head
            'use_cross_scale': True
        }
    }
    
    return ViTEnhancedUNet64(**configs[config])


# Drop-in replacement usage:
# model_edm = create_vit_enhanced_unet('balanced')
# # Your existing training loop works unchanged!