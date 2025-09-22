# model_sr3_vit.py
"""
   NOTE -- This model is not yet functioning.
"""

"""
Hybrid ViT-UNet for Super-Resolution Diffusion Models.

Author: DeepSeek

Design: This model integrates a Vision Transformer (ViT) encoder for processing the
low-resolution (64x64) conditioning image with a UNet for iterative denoising of the
high-resolution (256x256) output. The ViT captures global structural relationships
(e.g., character pose, limb connections) through self-attention, while the UNet handles
local detail synthesis. Global context is injected into the UNet decoder via cross-attention
at strategic points (bottleneck and deep decoder layers) to ensure coherent generation.

Key features for 6GB VRAM training:
- ViT processes only the 64x64 LR image, minimizing memory usage
- Cross-attention limited to bottleneck and low-resolution decoder layers
- Efficient channel dimensions and linear projections
- einx used for expressive tensor operations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einx

class ViTEncoder(nn.Module):
    """Vision Transformer encoder for global context extraction from LR images.
    
    Processes 64x64 images into token sequences that capture global structural
    relationships through self-attention mechanism.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, 
                 embed_dim=512, depth=6, n_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim) * 0.02
        )
        
        # Transformer blocks with LayerNorm, attention, and MLP
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, n_heads, batch_first=True),
                'norm2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                )
            }) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """Input: (B, C, H, W) low-res image. Output: (B, N_tokens, embed_dim) context tokens."""
        # Extract patch embeddings
        x = self.patch_embed(x)  # (B, E, H//P, W//P)
        x = einx.rearrange('b e h w -> b (h w) e', x)
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks with residual connections
        for block in self.blocks:
            # Self-attention
            resid = x
            x = block['norm1'](x)
            attn_out, _ = block['attn'](x, x, x)
            x = resid + attn_out
            
            # MLP
            resid = x
            x = block['norm2'](x)
            x = resid + block['mlp'](x)
        
        return self.norm(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for injecting ViT context into UNet features.
    
    Allows UNet decoder features to attend to global context from ViT tokens,
    enabling coherent structural generation.
    """
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """x: (B, C, H, W) UNet features, context: (B, N, D) ViT tokens"""
        B, C, H, W = x.shape
        
        # Rearrange UNet features for attention
        x_flat = einx.rearrange('b c h w -> b (h w) c', x)
        
        # Compute queries, keys, values
        q = self.to_q(x_flat)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        # Multi-head attention
        q, k, v = [einx.rearrange('b n (h d) -> (b h) n d', t, h=self.heads) 
                   for t in [q, k, v]]
        
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reassemble and project output
        attn_output = einx.rearrange('(b h) n d -> b n (h d)', attn_output, h=self.heads)
        attn_output = self.to_out(attn_output)
        
        # Reshape back to spatial dimensions and add residual
        attn_output = einx.rearrange('b (h w) c -> b c h w', attn_output, h=H, w=W)
        return x + attn_output


class ResnetBlock(nn.Module):
    """Residual block with optional time conditioning for diffusion models."""
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        
        self.mlp = None
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
        
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.res_conv = (nn.Conv2d(dim, dim_out, 1) 
                        if dim != dim_out else nn.Identity())

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = einx.rearrange('b c -> b c 1 1', time_emb)
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        
        h = self.block2(h)
        return h + self.res_conv(x)


class UNetSR3_ViT(nn.Module):
    """Hybrid ViT-UNet for 64x64→256x256 super-resolution diffusion.
    
    Architecture:
    1. ViT encoder processes 64x64 LR image to global context tokens
    2. UNet denoises 256x256 HR image with cross-attention to ViT context
    3. Strategic cross-attention at bottleneck and deep decoder layers
    
    Designed for 6GB VRAM training with balanced global coherence and local detail.
    """
    
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), channels=3,
                 vit_embed_dim=512, vit_depth=6, vit_heads=8,
                 cross_attention_layers=(1, 2)):  # Decoder indices for cross-attn
        super().__init__()
        
        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # ViT encoder for global context
        self.vit_encoder = ViTEncoder(
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            n_heads=vit_heads
        )
        self.vit_proj = nn.Linear(vit_embed_dim, dim)
        
        # Initial convolution (concatenates noisy_hr + upscaled_lr)
        self.init_conv = nn.Conv2d(channels * 2, dim, 7, padding=3)
        
        # Dimensions for encoder/decoder
        dims = [dim, *[dim * m for m in dim_mults]]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Create encoder blocks
        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i+1]
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1)  # Downsample
            ]))
        
        # Bottleneck
        bottleneck_dim = dims[-1]
        self.mid_block1 = ResnetBlock(bottleneck_dim, bottleneck_dim, time_emb_dim=time_dim)
        self.mid_cross_attn = CrossAttentionBlock(bottleneck_dim, dim)  # Global planning
        self.mid_block2 = ResnetBlock(bottleneck_dim, bottleneck_dim, time_emb_dim=time_dim)
        
        # Create decoder blocks with cross-attention at specified layers
        for i in range(len(dims) - 1):
            idx = len(dims) - 2 - i  # Reverse order
            dim_in = dims[idx + 1]
            dim_out = dims[idx]
            has_cross_attn = i in cross_attention_layers
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_in * 2, dim_out, time_emb_dim=time_dim),  # Skip connection
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                CrossAttentionBlock(dim_out, dim) if has_cross_attn else nn.Identity(),
                nn.ConvTranspose2d(dim_out, dim_out, 4, stride=2, padding=1)  # Upsample
            ]))
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, channels, 1)
        )

    def forward(self, x, time, x_lr):
        """
        Args:
            x: Noisy HR image [B, C, 256, 256]
            time: Diffusion timestep [B]
            x_lr: Low-res conditioning image [B, C, 64, 64]
        
        Returns:
            Predicted noise [B, C, 256, 256]
        """
        # Get global context from ViT
        vit_tokens = self.vit_encoder(x_lr)
        vit_context = self.vit_proj(vit_tokens)
        
        # Prepare UNet input
        x_lr_upscaled = F.interpolate(x_lr, size=x.shape[2:], mode='bilinear')
        x = torch.cat([x, x_lr_upscaled], dim=1)
        x = self.init_conv(x)
        residual = x
        
        # Time embedding
        t = self.time_mlp(time)
        
        # Encoder
        skips = []
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            skips.append(x)
            x = downsample(x)
        
        # Bottleneck with cross-attention
        x = self.mid_block1(x, t)
        x = self.mid_cross_attn(x, vit_context)  # Inject global context
        x = self.mid_block2(x, t)
        
        # Decoder
        for block1, block2, cross_attn, upsample in self.ups:
            # Get skip connection from encoder
            skip = skips.pop()
            
            # Ensure dimensions match for concatenation
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            
            x = torch.cat([x, skip], dim=1)  # Skip connection
            x = block1(x, t)
            x = block2(x, t)
            x = cross_attn(x, vit_context)  # Layer-specific context injection
            x = upsample(x)
        
        # Final residual connection and output
        x = x + residual
        return self.final_conv(x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# Sanity check the model's inputs and outputs are as expected
if __name__ == "__main__":
    B = 4
    hrimg = torch.randn(B, 3, 256, 256)
    lrimg = torch.randn(B, 3, 64, 64)
    t = torch.rand(B)

    # Test the model
    model = UNetSR3_ViT(dim=64)  # Smaller dim for testing
    
    with torch.no_grad():
        out = model(hrimg, t, lrimg)
    
    print("Input noisy HR shape:", hrimg.shape)  # should be [4, 3, 256, 256]
    print("Input LR shape:", lrimg.shape)        # should be [4, 3, 64, 64]
    print("Time shape:", t.shape)                # should be [4]
    print("Output shape:", out.shape)            # should be [4, 3, 256, 256]
    
    # Test ViT encoder separately
    vit_tokens = model.vit_encoder(lrimg)
    print("ViT tokens shape:", vit_tokens.shape) # should be [4, 65, 512] (64 patches + 1 CLS token)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"UNetSR3_ViT number of parameters: {num_params:,}")
    
    # Test that output is same dtype as input
    print(f"Output dtype: {out.dtype}, matches input: {out.dtype == hrimg.dtype}")