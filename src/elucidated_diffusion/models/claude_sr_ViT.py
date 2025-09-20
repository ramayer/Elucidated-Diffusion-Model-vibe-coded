import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# https://claude.ai/public/artifacts/240eea0b-6a2d-4f59-a343-772dc3dfa75b

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)) 
        )
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class PatchEmbedding(nn.Module):
    """Convert 64x64 image to sequence of patch embeddings"""
    def __init__(self, img_size=64, patch_size=4, in_ch=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 16x16 = 256 patches
        
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Project to patches: [B, embed_dim, 16, 16] -> [B, embed_dim, 256]
        x = self.proj(x).flatten(2)
        # Transpose: [B, 256, embed_dim]  
        x = x.transpose(1, 2)
        # Add positional encoding
        x = x + self.pos_embed
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class StructureViT(nn.Module):
    """Vision Transformer for extracting global structure from LR conditioning"""
    def __init__(self, img_size=64, patch_size=4, in_ch=3, embed_dim=384, 
                 depth=6, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        self.pos_drop = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale output projections for cross-attention
        self.to_32x32 = nn.Linear(embed_dim, 128)  # For 32x32 cross-attention
        self.to_64x64 = nn.Linear(embed_dim, 256)  # For 64x64 cross-attention
        self.to_16x16 = nn.Linear(embed_dim, 512)  # For 16x16 cross-attention
        
    def forward(self, x):
        # x: [B, 3, 64, 64]
        x = self.patch_embed(x)  # [B, 256, 384]
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)  # [B, 256, 384] - global structure features
        
        # Create multi-scale conditioning features
        # Reshape patch tokens back to spatial layout: 16x16 grid
        B, N, C = x.shape
        patch_size = int(N ** 0.5)  # 16
        
        # Project to different channel dimensions for multi-scale fusion
        feat_32 = self.to_32x32(x)  # [B, 256, 128]
        feat_64 = self.to_64x64(x)  # [B, 256, 256] 
        feat_16 = self.to_16x16(x)  # [B, 256, 512]
        
        # Reshape to spatial format for cross-attention
        feat_32 = feat_32.transpose(1, 2).view(B, 128, patch_size, patch_size)  # [B, 128, 16, 16]
        feat_64 = feat_64.transpose(1, 2).view(B, 256, patch_size, patch_size)  # [B, 256, 16, 16]
        feat_16 = feat_16.transpose(1, 2).view(B, 512, patch_size, patch_size)  # [B, 512, 16, 16]
        
        return {
            '16x16': feat_16,  # [B, 512, 16, 16] - for bottleneck
            '32x32': feat_32,  # [B, 128, 16, 16] - will upsample to 32x32
            '64x64': feat_64,  # [B, 256, 16, 16] - will upsample to 64x64
        }

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.block1(x))
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.block2(h))
        return h + self.res_conv(x)

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between UNet features and ViT conditioning features.
    
    Strengths:
    - Flexible attention patterns - can learn complex spatial relationships
    - Theoretically most expressive fusion method
    
    Weaknesses:  
    - Many parameters (~50K per fusion point) - slow to train
    - Complex optimization landscape - often fails to learn meaningful patterns
    - Requires learning both WHAT to attend to and HOW to attend
    - Memory intensive due to attention matrix computation
    """
    def __init__(self, unet_dim, vit_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (unet_dim // num_heads) ** -0.5
        
        self.to_q = nn.Conv2d(unet_dim, unet_dim, 1)
        self.to_kv = nn.Conv2d(vit_dim, unet_dim * 2, 1)
        self.to_out = nn.Conv2d(unet_dim, unet_dim, 1)
        
        self.norm_unet = nn.GroupNorm(8, unet_dim)
        self.norm_vit = nn.GroupNorm(8, vit_dim)

    def forward(self, unet_feat, vit_feat):
        B, C_u, H_u, W_u = unet_feat.shape
        B, C_v, H_v, W_v = vit_feat.shape
        
        # Upsample ViT features to match UNet resolution if needed
        if (H_v, W_v) != (H_u, W_u):
            vit_feat = F.interpolate(vit_feat, size=(H_u, W_u), mode='bilinear', align_corners=False)
        
        # Normalize
        unet_norm = self.norm_unet(unet_feat)
        vit_norm = self.norm_vit(vit_feat)
        
        # Compute Q, K, V
        q = self.to_q(unet_norm).view(B, C_u, -1)  # [B, C_u, H_u*W_u]
        kv = self.to_kv(vit_norm).view(B, C_u * 2, -1)  # [B, C_u*2, H_u*W_u]
        k, v = kv.chunk(2, dim=1)  # Each: [B, C_u, H_u*W_u]
        
        # Multi-head attention
        q = q.view(B, self.num_heads, C_u // self.num_heads, -1).transpose(-2, -1)
        k = k.view(B, self.num_heads, C_u // self.num_heads, -1).transpose(-2, -1) 
        v = v.view(B, self.num_heads, C_u // self.num_heads, -1).transpose(-2, -1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(-2, -1)
        out = out.contiguous().view(B, C_u, H_u, W_u)
        out = self.to_out(out)
        # see https://claude.ai/share/64c68f34-8d47-4753-839a-7a24da388c85
        #     with torch.no_grad():
        #         attn_detached = attn.detach().cpu()
        #         entropy = -(attn_detached * attn_detached.log()).sum(dim=-1).mean().item()
        #         print(f"Attention entropy (step {self.debug_counter}): {entropy}")
        return unet_feat + out


class SimpleFusion(nn.Module):
    """
    Simple addition-based fusion with 1x1 projection.
    
    Strengths:
    - Fast training - direct information flow with no learning barriers
    - Few parameters (~1K per fusion point) - memory efficient
    - Stable optimization - just learns a projection matrix
    - Preserves spatial correspondence perfectly
    
    Weaknesses:
    - No spatial context - each UNet region only sees its own ViT patch
    - May struggle with fine details requiring neighbor information
    - Limited expressiveness compared to attention mechanisms
    """
    def __init__(self, unet_dim, vit_dim):
        super().__init__()
        self.proj = nn.Conv2d(vit_dim, unet_dim, kernel_size=1)
        
    def forward(self, unet_feat, vit_feat):
        """
        Args:
            unet_feat: UNet features [B, unet_dim, H, W]
            vit_feat: ViT features [B, vit_dim, H_v, W_v]
        Returns:
            Fused features [B, unet_dim, H, W]
        """
        B, C_u, H_u, W_u = unet_feat.shape
        B, C_v, H_v, W_v = vit_feat.shape
        
        # Upsample ViT features to match UNet spatial resolution if needed
        if (H_v, W_v) != (H_u, W_u):
            vit_resized = F.interpolate(vit_feat, size=(H_u, W_u), mode='nearest')
        else:
            vit_resized = vit_feat
            
        # Project ViT features to match UNet channel dimension
        vit_proj = self.proj(vit_resized)
        
        # Simple addition fusion
        return unet_feat + vit_proj

class NeighborAwareFusion(nn.Module):
    """
    Addition-based fusion with 3x3 projection for spatial context.
    
    Strengths:
    - Fast training like SimpleFusion but with spatial awareness
    - Each UNet region sees its ViT patch + 8 neighbors
    - Good for fine details requiring local spatial relationships (e.g., faces)
    - Still lightweight (~9K parameters per fusion point)
    
    Weaknesses:  
    - Only sees immediate neighbors - limited spatial range
    - More parameters than SimpleFusion (but far fewer than CrossAttention)
    - May still struggle with very long-range spatial dependencies

    Details:
    - Parameter count: 9K (vs SimpleFusion's 1K vs CrossAttention's 50K+)
    - Spatial awareness: 3×3 receptive field per region
    - Training speed: Should be nearly as fast as SimpleFusion

    For faces, that 3×3 neighborhood might be exactly what you need - a "left eye" region 
    can now see information from "nose", "right eye", and "forehead" patches to help 
    with precise facial geometry.
    """
    def __init__(self, unet_dim, vit_dim):
        super().__init__()
        # 3x3 conv gives each region access to its patch + 8 neighbors
        self.proj = nn.Conv2d(vit_dim, unet_dim, kernel_size=3, padding=1)
        # Smaller initial weights for the 3x3 conv: nn.init.xavier_normal_(self.proj.weight, gain=0.1)
        nn.init.xavier_normal_(self.proj.weight, gain=0.1)
        
    def forward(self, unet_feat, vit_feat):
        B, C_u, H_u, W_u = unet_feat.shape
        B, C_v, H_v, W_v = vit_feat.shape
        
        # Upsample ViT features to match UNet spatial resolution
        if (H_v, W_v) != (H_u, W_u):
            vit_resized = F.interpolate(vit_feat, size=(H_u, W_u), mode='nearest')
        else:
            vit_resized = vit_feat
            
        # Project with 3x3 conv (spatial context) and add
        vit_proj = self.proj(vit_resized)  # Each pixel sees neighbors
        return unet_feat + vit_proj

class HybridViTUNetSR(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, time_emb_dim=128,
                 fusion_style="SimpleFusion"):
        super().__init__()

        self.fusion_style = fusion_style
        
        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # ViT for LR structure understanding
        self.structure_vit = StructureViT(img_size=64, embed_dim=384, depth=6)
        
        # UNet for HR processing
        self.enc1 = ResBlock(in_ch, base_ch, time_emb_dim)      # 256x256
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)  # 128x128
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)  # 64x64
        self.enc4 = ResBlock(base_ch * 4, base_ch * 8, time_emb_dim)  # 32x32
        
        # Cross-attention fusion modules
        if self.fusion_style=="CrossAttention":
            self.fusion_64 = CrossAttentionFusion(base_ch * 4, 256)  # 64x64 level
            self.fusion_32 = CrossAttentionFusion(base_ch * 8, 128)  # 32x32 level  
            self.fusion_16 = CrossAttentionFusion(base_ch * 8, 512)  # 16x16 bottleneck
        elif self.fusion_style=="NeighborAwareFusion":
            self.fusion_64 = NeighborAwareFusion(base_ch * 4, 256)  # 64x64 level
            self.fusion_32 = NeighborAwareFusion(base_ch * 8, 128)  # 32x32 level  
            self.fusion_16 = NeighborAwareFusion(base_ch * 8, 512)  # 16x16 bottleneck
        else:
            self.fusion_64 = SimpleFusion(base_ch * 4, 256)  # 64x64 level
            self.fusion_32 = SimpleFusion(base_ch * 8, 128)  # 32x32 level  
            self.fusion_16 = SimpleFusion(base_ch * 8, 512)  # 16x16 bottleneck
        
        # Bottleneck
        self.mid = ResBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        
        # Decoder
        self.dec4 = ResBlock(base_ch * 8 + base_ch * 8, base_ch * 4, time_emb_dim)
        self.dec3 = ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, time_emb_dim)
        self.dec2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, time_emb_dim)
        self.dec1 = ResBlock(base_ch + base_ch, base_ch, time_emb_dim)
        
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)
        
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x_noisy, lr_img, t):
        """
        x_noisy: [B, 3, 256, 256] - noisy HR image
        lr_img: [B, 3, 64, 64] - LR conditioning image
        t: [B] - timesteps
        """
        # Time embedding
        t_emb = self.time_emb(t)
        
        # Extract global structure from LR using ViT
        structure_feats = self.structure_vit(lr_img)
        
        # UNet encoder path  
        e1 = self.enc1(x_noisy, t_emb)         # [B, 64, 256, 256]
        e2 = self.enc2(self.down(e1), t_emb)   # [B, 128, 128, 128]
        e3 = self.enc3(self.down(e2), t_emb)   # [B, 256, 64, 64]
        e4 = self.enc4(self.down(e3), t_emb)   # [B, 512, 32, 32]
        
        # Fuse UNet features with ViT structure features, optionally via cross-attention
        e3_fused = self.fusion_64(e3, structure_feats['64x64'])  # Global structure -> local 64x64
        e4_fused = self.fusion_32(e4, structure_feats['32x32'])  # Global structure -> local 32x32

        # Bottleneck with structure fusion
        m = self.mid(self.down(e4_fused), t_emb)  # [B, 512, 16, 16]
        m_fused = self.fusion_16(m, structure_feats['16x16'])  # Global structure -> bottleneck
        
        # Decoder path
        d4 = self.dec4(torch.cat([self.up(m_fused), e4_fused], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up(d4), e3_fused], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
        
        return self.out_conv(d1)


# Test the architecture
if __name__ == "__main__":
    model = HybridViTUNetSR()
    
    B = 2
    x_noisy = torch.randn(B, 3, 256, 256)
    lr_img = torch.randn(B, 3, 64, 64)
    t = torch.rand(B)
    
    with torch.no_grad():
        out = model(x_noisy, lr_img, t)
    
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Break down parameter count
    vit_params = sum(p.numel() for p in model.structure_vit.parameters())
    unet_params = sum(p.numel() for p in model.parameters()) - vit_params
    print(f"ViT parameters: {vit_params:,}")
    print(f"UNet + fusion parameters: {unet_params:,}")
