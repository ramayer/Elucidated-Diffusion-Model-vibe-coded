import torch
import torch.nn.functional as F

# EDM noise schedule, loss weighting, and sampling procedure
# EDM parameters from NVIDIA's reference
P_mean = -1.2  # Mean of log-normal distribution for sigma sampling
P_std = 1.2    # Std of log-normal distribution for sigma sampling
sigma_data = 0.5  # Data standard deviation
sigma_min = 0.002
sigma_max = 80
rho = 7

# EDM noise schedule (sampling)
def edm_sigma_schedule(t):
    return (sigma_max ** (1/rho) + t * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho

# EDM loss weighting - CORRECTED according to NVIDIA's implementation
# weight = (σ² + σ_data²) / (σ × σ_data)²
def edm_loss_weight(sigma, sigma_data=sigma_data):
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


if show_edm_schedule:=False:
    # Example: plot the EDM noise schedule
    ts = torch.linspace(0, 1, 100)
    sigmas = edm_sigma_schedule(ts)
    plt.plot(ts.numpy(), sigmas.numpy())
    plt.xlabel('t')
    plt.ylabel('sigma(t)')
    plt.title('EDM Noise Schedule')
    plt.show()

    # or easier: 

    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(x, edm_sigma_schedule(x))


# TODO - merge these --- they're so close but so different.

# EDM sampling procedure - Following NVIDIA's edm_sampler (Algorithm 2)
def edm_ancestral_sampling_for_diffusion(model, num_steps=18, batch_size=8, img_shape=(3, 64, 64)):
    device = next(model.parameters()).device
    
    # Initialize noise
    x_next = torch.randn((batch_size,) + img_shape, device=device)
    
    # Time step schedule (matching NVIDIA's implementation)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * 
               (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    
    # Initialize with first noise level
    x_next = x_next * t_steps[0]
    
    # Main sampling loop (Heun's method)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Preconditioning coefficients for current timestep
        sigma = t_cur.float()
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # Euler step
        F_x = model(c_in * x_cur, c_noise.expand(batch_size))
        denoised = c_skip * x_cur + c_out * F_x
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        # Apply 2nd order correction (Heun's method) - except for last step
        if i < num_steps - 1:
            # Preconditioning coefficients for next timestep
            sigma_next = t_next.float()
            c_skip_next = sigma_data ** 2 / (sigma_next ** 2 + sigma_data ** 2)
            c_out_next = sigma_next * sigma_data / (sigma_next ** 2 + sigma_data ** 2).sqrt()
            c_in_next = 1 / (sigma_data ** 2 + sigma_next ** 2).sqrt()
            c_noise_next = sigma_next.log() / 4
            
            F_x_next = model(c_in_next * x_next, c_noise_next.expand(batch_size))
            denoised_next = c_skip_next * x_next + c_out_next * F_x_next
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next.to("cpu")



# TODO - re-merge with the one above
def edm_ancestral_sampling_for_sr(model, lr_64, num_steps=18, batch_size=8, img_shape=(1, 28, 28),
                           headstart_sigma:float|None=None):
    """
        EDM ancestral sampling with optional noisy-LR head-start.
        
        Args:
            model: UNet denoiser
            lr_up: upscaled LR image [B, 3, H, W]
            num_steps: number of sampling steps
            batch_size: batch size
            img_shape: shape of output images
            headstart_sigma: float or None
                If None, start from pure noise.
                If float, adds Gaussian noise with this std to lr_up for head-start.
    """
    device = next(model.parameters()).device
    lr_up = F.interpolate(lr_64, size=(256,256), mode='bilinear', align_corners=False)

    # Initialize noise
    # ------------------- Head-start initialization -------------------
    if headstart_sigma is None:
        # Original behavior: pure Gaussian noise
        x_next = torch.randn((batch_size,) + img_shape, device=device)
    else:
        # Start from LR upsample + moderate noise
        lr_up_resized = F.interpolate(lr_up, size=img_shape[-2:], mode='bilinear', align_corners=False)
        x_next = lr_up_resized.to(device) + headstart_sigma * torch.randn((batch_size,) + img_shape, device=device)
    # ------------------------------------------------------------------

    
    # Time step schedule (matching NVIDIA's implementation)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * 
               (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    
    # Initialize with first noise level
    #-----------------------------------------------------
    # after head-start initialization
    if headstart_sigma is not None:
        """
        https://chatgpt.com/share/68c6111a-55d0-800b-b1b3-14e3cf3ff731
        Some papers also rescale the remaining t_steps proportionally to account for the lower starting noise, 
        so the schedule adapts dynamically. But the simplest correct approach is just to replace the first t_steps[0] with the actual headstart_sigma.
        """
        #t_steps[0] = headstart_sigma
        #sigma_max = headstart_sigma
        t_steps = (headstart_sigma ** (1/rho) + step_indices / (num_steps - 1) * 
               (sigma_min ** (1/rho) - headstart_sigma ** (1/rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    else:
        x_next = x_next * t_steps[0]

    
    #-----------------------------------------------------
    # Main sampling loop (Heun's method)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Preconditioning coefficients for current timestep
        sigma = t_cur.float()
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
        c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # Euler step
        this_models_input = torch.cat([c_in*x_cur, lr_up], dim=1).to(device)
        #F_x = model(c_in * x_cur, c_noise.expand(batch_size))
        F_x = model(this_models_input, c_noise.expand(batch_size))
        denoised = c_skip * x_cur + c_out * F_x
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        
        # Apply 2nd order correction (Heun's method) - except for last step
        if i < num_steps - 1:
            # Preconditioning coefficients for next timestep
            sigma_next = t_next.float()
            c_skip_next = sigma_data ** 2 / (sigma_next ** 2 + sigma_data ** 2)
            c_out_next = sigma_next * sigma_data / (sigma_next ** 2 + sigma_data ** 2).sqrt()
            c_in_next = 1 / (sigma_data ** 2 + sigma_next ** 2).sqrt()
            c_noise_next = sigma_next.log() / 4
            
            #F_x_next = model(c_in_next * x_next, c_noise_next.expand(batch_size))
            this_models_input = torch.cat([c_in_next * x_next, lr_up], dim=1).to(device)
            F_x_next = model(this_models_input, c_noise_next.expand(batch_size))


            denoised_next = c_skip_next * x_next + c_out_next * F_x_next
            d_prime = (x_next - denoised_next) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    
    return x_next.to("cpu")
