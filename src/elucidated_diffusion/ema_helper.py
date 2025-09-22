import copy
import torch
import math
class EMAHelper:
    """
    EMAHelper implements Exponential Moving Average (EMA) tracking for PyTorch models.
    This helper class maintains a frozen copy of a given model and updates its parameters
    using an exponential moving average of the original model's parameters. The EMA decay
    rate can be dynamically adjusted based on the training step, allowing for smooth
    transition from a lower to a higher decay value.
    EMAHelper is generic and can be used with any PyTorch model, including both Diffusion Models
    and SuperResolution Models within this project. It is designed to improve model stability
    and performance during training by providing a smoothed version of the model for evaluation
    or inference.
    Args:
        model (torch.nn.Module): The model to track with EMA.
        d_final (float, optional): The final EMA decay rate. Default is 0.999.
    Attributes:
        ema_model (torch.nn.Module): The EMA-tracked copy of the model (in eval mode).
        step (int): The current training step for dynamic decay calculation.
    Methods:
        get_ema_decay(): Computes the current EMA decay rate.
        update(model): Updates the EMA model parameters using the current model.
        get_model(): Returns the EMA-smoothed model (in eval mode).
    """

    def __init__(self, model, step=1, d_final=0.999):
        # Make a frozen copy of the model for EMA tracking
        self.ema_model = copy.deepcopy(model).eval()
        self.step = step
        self.d_final = d_final
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def get_ema_decay(self, step = None):
        """
        Progressive EMA decay that starts loose and tightens,
        but always slightly favors newer weights.
        """
        step = step or self.step
        # Start with short effective window ~ step
        d = 1 - 1/(step+1)
        # Gradually push it closer to base as step grows
        return d * self.d_final

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        decay = self.get_ema_decay()
        for k, v in self.ema_model.state_dict().items():
            if k in msd:
                v.copy_(decay * v + (1 - decay) * msd[k])
        self.step += 1

    def get_model(self):
        """Return the EMA-smoothed model (already in eval mode)."""
        return self.ema_model



