import torch
import pytest
from elucidated_diffusion.ema_helper import EMAHelper

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

def test_ema_initialization_and_get_model():
    model = TinyModel()
    ema = EMAHelper(model)
    ema_model = ema.get_model()
    assert isinstance(ema_model, TinyModel)
    for p in ema_model.parameters():
        assert not p.requires_grad
    assert not ema_model.training  # Should be in eval mode

def test_ema_update_moves_towards_model():
    model = TinyModel()
    # Set model weights to all ones
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
    ema = EMAHelper(model)
    # Set EMA model weights to all zeros
    with torch.no_grad():
        for p in ema.ema_model.parameters():
            p.fill_(0.0)
    # Step 0: decay is near 0.9, so after update, EMA weights should be ~0.1
    ema.update(model)
    for p in ema.ema_model.parameters():
        # Should be close to (decay * 0 + (1-decay) * 1) = (1-decay)
        expected = 1 - ema.get_ema_decay(step=1)
        assert torch.allclose(p, torch.full_like(p, expected), atol=1e-6)

def test_ema_decay_increases_with_step():
    model = TinyModel()
    ema = EMAHelper(model)
    d1 = ema.get_ema_decay()
    ema.step = 10000
    d2 = ema.get_ema_decay()
    assert d2 > d1
    assert d2 <= 0.999

def test_ema_update_multiple_steps():
    model = TinyModel()
    ema = EMAHelper(model)
    # Set model weights to 1, EMA weights to 0
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.0)
        for p in ema.ema_model.parameters():
            p.fill_(0.0)
    # Run several EMA updates
    for i in range(5):
        ema.update(model)
    # EMA weights should be between 0 and 1
    for p in ema.ema_model.parameters():
        assert torch.all(p >= 0)
        assert torch.all(p >= 0.8), "should approach the new model"
        assert torch.all(p <= 1)

def test_ema_update_with_identical_models():
    model = TinyModel()
    ema = EMAHelper(model)
    # Set both model and EMA model weights to 0.5
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.5)
        for p in ema.ema_model.parameters():
            p.fill_(0.5)
    ema.update(model)
    for p in ema.ema_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5), atol=1e-6)