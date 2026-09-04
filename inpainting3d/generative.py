from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
ObjectiveName = Literal["diffusion", "flow_matching"]


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Build sinusoidal embeddings for continuous time values in [0, 1]."""
    if t.ndim == 0:
        t = t[None]
    t = t.float().view(-1, 1) * 1000.0
    half = dim // 2
    if half <= 1:
        freqs = torch.ones(1, device=t.device, dtype=t.dtype)
    else:
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1)
        )
    args = t * freqs.view(1, -1)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


class TimeEmbedding(nn.Module):
    """Small MLP that maps scalar diffusion or flow time to a feature vector."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = int(hidden_dim or dim * 4)
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.out_dim = hidden

    def forward(self, t: Tensor) -> Tensor:
        return self.net(sinusoidal_time_embedding(t, self.dim))


class TimeResidualBlock3D(nn.Module):
    """Residual 3D block with additive time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.dropout = nn.Dropout3d(float(dropout)) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(time_emb).view(time_emb.shape[0], -1, 1, 1, 1)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.shortcut(x))


class TimeConditionedUNet3D(nn.Module):
    """
    Compact conditional 3D U-Net for diffusion and flow-matching inpainting.

    Input channels are expected to be `[latent_or_noisy_volume, sparse_volume,
    known_mask]`. The same network can predict DDPM noise or a flow-matching
    velocity field depending on the training objective.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        init_feat: int = 16,
        levels: int = 4,
        time_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")

        self.time = TimeEmbedding(time_dim)
        time_out = self.time.out_dim
        features = [int(init_feat) * (2 ** i) for i in range(int(levels))]

        self.encoders = nn.ModuleList()
        current = int(in_channels)
        for feat in features:
            self.encoders.append(TimeResidualBlock3D(current, feat, time_out, dropout=dropout))
            current = feat

        self.pool = nn.MaxPool3d(2)
        self.bottleneck = TimeResidualBlock3D(features[-1], features[-1] * 2, time_out, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(current, feat, kernel_size=2, stride=2))
            self.decoders.append(TimeResidualBlock3D(feat * 2, feat, time_out, dropout=dropout))
            current = feat

        self.final = nn.Conv3d(features[0], int(out_channels), kernel_size=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        time_emb = self.time(t.to(device=x.device))
        skips: List[Tensor] = []
        h = x
        for block in self.encoders:
            h = block(h, time_emb)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h, time_emb)

        for up, block in zip(self.upconvs, self.decoders):
            h = up(h)
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1), time_emb)
        return self.final(h)


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: Tensor
    alphas: Tensor
    alpha_cumprod: Tensor


def make_diffusion_schedule(
    num_steps: int,
    device: torch.device,
    schedule: Literal["cosine", "linear"] = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> DiffusionSchedule:
    """
    Create a DDPM noise schedule.

    The cosine schedule follows the commonly used improved-DDPM variant and is
    a strong default for image/volume diffusion.
    """
    steps = int(num_steps)
    if steps < 2:
        raise ValueError("num_steps must be >= 2")

    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, steps, device=device)
    elif schedule == "cosine":
        s = 0.008
        x = torch.linspace(0, steps, steps + 1, device=device)
        alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, min=1e-6, max=0.999)
    else:
        raise ValueError(f"Unknown diffusion schedule: {schedule}")

    alphas = 1.0 - betas
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_cumprod=torch.cumprod(alphas, dim=0))


def _extract(values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
    out = values.gather(0, timesteps.long())
    return out.view(timesteps.shape[0], *([1] * (target.ndim - 1)))


def normalize_timesteps(timesteps: Tensor, num_steps: int) -> Tensor:
    return timesteps.float() / float(max(1, int(num_steps) - 1))


def q_sample(x0: Tensor, timesteps: Tensor, noise: Tensor, schedule: DiffusionSchedule) -> Tensor:
    alpha_bar = _extract(schedule.alpha_cumprod, timesteps, x0)
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise


def weighted_mse(pred: Tensor, target: Tensor, known_mask: Tensor, hole_weight: float = 1.0, valid_weight: float = 1.0) -> Tensor:
    weights = known_mask.float() * float(valid_weight) + (1.0 - known_mask.float()) * float(hole_weight)
    return (((pred - target) ** 2) * weights).sum() / weights.sum().clamp_min(1e-8)


def diffusion_training_loss(
    model: nn.Module,
    x0: Tensor,
    sparse: Tensor,
    known_mask: Tensor,
    schedule: DiffusionSchedule,
    hole_weight: float = 1.0,
    valid_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    batch = x0.shape[0]
    timesteps = torch.randint(0, schedule.betas.numel(), (batch,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, timesteps, noise, schedule)
    t_norm = normalize_timesteps(timesteps, schedule.betas.numel())
    pred_noise = model(torch.cat([xt, sparse, known_mask], dim=1), t_norm)
    loss = weighted_mse(pred_noise, noise, known_mask, hole_weight=hole_weight, valid_weight=valid_weight)
    return loss, {"t_mean": float(t_norm.mean().detach().cpu().item())}


def flow_matching_training_loss(
    model: nn.Module,
    x0: Tensor,
    sparse: Tensor,
    known_mask: Tensor,
    hole_weight: float = 1.0,
    valid_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    batch = x0.shape[0]
    t = torch.rand(batch, device=x0.device, dtype=x0.dtype)
    noise = torch.randn_like(x0)
    t_view = t.view(batch, 1, 1, 1, 1)
    xt = (1.0 - t_view) * noise + t_view * x0
    target_velocity = x0 - noise
    pred_velocity = model(torch.cat([xt, sparse, known_mask], dim=1), t)
    loss = weighted_mse(pred_velocity, target_velocity, known_mask, hole_weight=hole_weight, valid_weight=valid_weight)
    return loss, {"t_mean": float(t.mean().detach().cpu().item())}


def sampling_timesteps(num_train_steps: int, sample_steps: int) -> List[int]:
    steps = min(max(1, int(sample_steps)), int(num_train_steps))
    raw = torch.linspace(int(num_train_steps) - 1, 0, steps).round().long().tolist()
    out: List[int] = []
    for value in raw:
        value_i = int(value)
        if not out or out[-1] != value_i:
            out.append(value_i)
    if out[-1] != 0:
        out.append(0)
    return out


@torch.no_grad()
def sample_diffusion_ddim(
    model: nn.Module,
    sparse: Tensor,
    known_mask: Tensor,
    schedule: DiffusionSchedule,
    sample_steps: int = 50,
    hard_constraint: bool = True,
) -> Tensor:
    model.eval()
    x = torch.randn_like(sparse)
    timesteps = sampling_timesteps(schedule.betas.numel(), sample_steps)
    for index, step in enumerate(timesteps):
        step_batch = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        t_norm = normalize_timesteps(step_batch, schedule.betas.numel())
        pred_noise = model(torch.cat([x, sparse, known_mask], dim=1), t_norm)
        alpha_bar_t = _extract(schedule.alpha_cumprod, step_batch, x)
        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t).clamp_min(1e-8)
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)
        if hard_constraint:
            x0_pred = known_mask * sparse + (1.0 - known_mask) * x0_pred
        if step == 0:
            x = x0_pred
            break
        next_step = timesteps[index + 1] if index + 1 < len(timesteps) else 0
        next_batch = torch.full((x.shape[0],), next_step, device=x.device, dtype=torch.long)
        alpha_bar_next = _extract(schedule.alpha_cumprod, next_batch, x)
        x = torch.sqrt(alpha_bar_next) * x0_pred + torch.sqrt(1.0 - alpha_bar_next) * pred_noise
    return torch.clamp(x, 0.0, 1.0)


@torch.no_grad()
def sample_flow_matching(
    model: nn.Module,
    sparse: Tensor,
    known_mask: Tensor,
    sample_steps: int = 50,
    hard_constraint: bool = True,
) -> Tensor:
    model.eval()
    steps = max(1, int(sample_steps))
    x = torch.randn_like(sparse)
    dt = 1.0 / float(steps)
    for step in range(steps):
        t = torch.full((x.shape[0],), float(step) / float(steps), device=x.device, dtype=x.dtype)
        velocity = model(torch.cat([x, sparse, known_mask], dim=1), t)
        x = x + dt * velocity
    x = torch.clamp(x, 0.0, 1.0)
    if hard_constraint:
        x = known_mask * sparse + (1.0 - known_mask) * x
    return x


def build_generative_model(
    init_feat: int = 16,
    levels: int = 4,
    time_dim: int = 64,
    dropout: float = 0.0,
) -> TimeConditionedUNet3D:
    return TimeConditionedUNet3D(
        in_channels=3,
        out_channels=1,
        init_feat=init_feat,
        levels=levels,
        time_dim=time_dim,
        dropout=dropout,
    )
