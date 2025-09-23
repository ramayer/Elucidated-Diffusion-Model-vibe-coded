import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper: sinusoidal timestep embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B] in [0,1] float
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)) 
        )
        emb = t[:, None] * freqs[None, :]   # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return emb

# --- Basic residual block ---
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # x: [B, C, H, W], t_emb: [B, time_emb_dim]
        h = F.silu(self.block1(x))
        # Add time embedding as bias
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = F.silu(self.block2(h))
        return h + self.res_conv(x)

# --- U-Net core ---
class UNetSR3(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResBlock(in_ch, base_ch, time_emb_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)

        # Bottleneck
        self.mid = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # Decoder
        self.dec3 = ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, time_emb_dim)
        self.dec2 = ResBlock(base_ch * 2 + base_ch * 2, base_ch, time_emb_dim)
        self.dec1 = ResBlock(base_ch + base_ch, base_ch, time_emb_dim)

        # Final output: predict noise (same shape as HR image)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x_noisy, lr_img, t):
        """
        x_noisy: [B, 3, 256, 256] - noisy HR image
        lr_img: [B, 3, 64, 64] - LR conditioning image
        t: [B] - timesteps
        """
        lr_up = F.interpolate(lr_img, size=x_noisy.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x_noisy, lr_up], dim=1) 
    #def forward(self, x, t):
    #    """
    #    x: [B, 6, 256, 256] = concat([x_noisy, lr_upsampled])
    #    t: [B] float timesteps normalized to [0,1]
    #    """
        # ---- time embedding ----
        t_emb = self.time_emb(t)  # [B, time_emb_dim]

        # ---- down path ----
        e1 = self.enc1(x, t_emb)           # [B, 64, 256, 256]
        e2 = self.enc2(self.down(e1), t_emb)  # [B, 128, 128, 128]
        e3 = self.enc3(self.down(e2), t_emb)  # [B, 256, 64, 64]

        # ---- bottleneck ----
        m = self.mid(e3, t_emb)             # [B, 256, 64, 64]

        # ---- up path ----
        #d3 = self.dec3(torch.cat([self.up(m), e3], dim=1), t_emb)  # [B, 128, 64, 64]
        #d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb) # [B, 64, 128, 128]
        #d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb) # [B, 64, 256, 256]
        # ---- up path (fixed spatial alignment) ----
        d3 = self.dec3(torch.cat([m, e3], dim=1), t_emb)            # [B, 128, 64, 64]
        d3_up = self.up(d3)                                         # [B, 128, 128, 128]

        d2 = self.dec2(torch.cat([d3_up, e2], dim=1), t_emb)         # [B, 64, 128, 128]
        d2_up = self.up(d2)                                         # [B, 64, 256, 256]

        d1 = self.dec1(torch.cat([d2_up, e1], dim=1), t_emb)         # [B, 64, 256, 256]

        return self.out_conv(d1)  # [B, 3, 256, 256]



# Sanity check the model's inpus and outputs are as expected
#   Add similar to the pytest for your project
if sanity_check_model := False:
    B = 4
    hrimg = torch.randn(B, 3, 256, 256).cuda()
    lrimg = torch.randn(B, 3, 64, 64).cuda()
    t = torch.rand(B).cuda()

    lr_up = F.interpolate(lrimg, size=(256,256), mode='bilinear', align_corners=False)
    x_noisy = torch.randn_like(hrimg)
    x_in = torch.cat([x_noisy, lr_up], dim=1).cuda()

    model = UNetSR3().cuda()
    with torch.no_grad():
        out = model(x_in, t)
    print("Output:", out.shape)  # should be [4, 3, 256, 256]
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SRUNet number of parameters: {num_params:,}")