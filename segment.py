"""Step-controlled 2D EMA segmentation for patient selection on ACDC."""

from copy import deepcopy
from dataclasses import dataclass
import os
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NUM_CLASSES = 4
EMA_MOMENTUM = 0.99
PSEUDO_THRESHOLD = 0.95


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(1, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.bottleneck = ConvBlock(32, 64)
        self.decoder2 = ConvBlock(64 + 32, 32)
        self.decoder1 = ConvBlock(32 + 16, 16)
        self.classifier = nn.Conv2d(16, NUM_CLASSES, 1)

    def forward(self, images):
        level1 = self.encoder1(images)
        level2 = self.encoder2(F.max_pool2d(level1, 2))
        features = self.bottleneck(F.max_pool2d(level2, 2))
        features = F.interpolate(features, size=level2.shape[-2:], mode="nearest")
        features = self.decoder2(torch.cat((features, level2), dim=1))
        features = F.interpolate(features, size=level1.shape[-2:], mode="nearest")
        features = self.decoder1(torch.cat((features, level1), dim=1))
        return self.classifier(features)


@dataclass
class TrainState:
    student: dict
    teacher: dict
    optimizer: dict
    prior: torch.Tensor
    steps: int
    seconds: float
    supervised_loss: float | None
    unlabeled_loss: float | None
    pseudo_fraction: float | None


def configure(device):
    """Use reproducible convolution settings and keep CPU runs responsive."""
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch environment.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device


class SlicePool:
    """Uniformly sample across slices without materializing duplicate pools."""

    def __init__(self, cases, ids):
        self.cases = [cases[index] for index in ids]
        self.ends = np.cumsum([len(case.images) for case in self.cases], dtype=np.int64)
        if self.cases and any(len(case.images) == 0 for case in self.cases):
            raise ValueError("Every eligible case must contain at least one image slice.")

    def sample(self, rng, count, *, with_masks):
        flat_ids = rng.integers(self.ends[-1], size=count)
        case_ids = np.searchsorted(self.ends, flat_ids, side="right")
        starts = np.where(case_ids == 0, 0, self.ends[case_ids - 1])
        slices = [
            (self.cases[case_id], slice_id)
            for case_id, slice_id in zip(case_ids, flat_ids - starts)
        ]
        images = np.stack([case.images[index] for case, index in slices])
        masks = np.stack([case.masks[index] for case, index in slices]) if with_masks else None
        return images, masks


def _image_tensor(images, device):
    return torch.from_numpy(images[:, None]).to(device, dtype=torch.float32)


def _supervised_loss(logits, masks):
    probs = logits.softmax(dim=1)[:, 1:]
    labels = F.one_hot(masks, NUM_CLASSES).movedim(-1, 1)[:, 1:]
    axes = (0, 2, 3)
    intersection = (probs * labels).sum(axes)
    denominator = probs.sum(axes) + labels.sum(axes)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return F.cross_entropy(logits, masks) + dice_loss


def _normalize_prior(prior):
    prior = prior.float().clamp_min(1e-8)
    return prior / prior.sum().clamp_min(1e-8)


@torch.no_grad()
def _diffunder_uot(probabilities, prior, masks):
    """Port the dense, all-valid-pixel path of DiffUnder train.py at be6f4d3.

    The current implementation relaxes both marginals. The publication's
    objective fixes the source marginal. This toggle targets current code.
    """
    histogram = _normalize_prior(torch.bincount(masks.flatten(), minlength=NUM_CLASSES).float())
    updated = _normalize_prior(prior * 0.99 + histogram * 0.01)
    prior.copy_(torch.where(masks.amax() > 0, updated, prior))
    foreground = probabilities[:, 1:].sum(dim=1).mean()
    relax = 0.1 + 9.9 * (foreground / 0.05).clamp(0, 1)
    exponent = relax / (relax + 0.5)
    flat = probabilities.movedim(1, -1).reshape(-1, NUM_CLASSES)
    transport = (flat.clamp_min(1e-6).log() / 0.5).exp().clamp_min(1e-8)
    row_prior = 1.0 / flat.shape[0]
    col_prior = _normalize_prior(prior)
    for _ in range(5):
        row_scale = (row_prior / transport.sum(dim=1, keepdim=True).clamp_min(1e-8)).pow(exponent)
        col_scale = (col_prior / transport.sum(dim=0).clamp_min(1e-8)).pow(exponent)
        transport = transport * row_scale * col_scale
    transport = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return transport.reshape(probabilities.movedim(1, -1).shape).movedim(-1, 1)


def _autocast(device):
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp)


def _model_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}


def _optimizer_state(optimizer):
    state = optimizer.state_dict()
    for values in state["state"].values():
        for name, value in values.items():
            if torch.is_tensor(value):
                values[name] = value.detach().cpu().clone()
    return state


def train(cases, labeled, unlabeled, *, steps: int, seed: int, device: str,
          start=None, batch_size: int = 4, uot: bool = False):
    """Train a fresh model or fork a checkpoint for exactly `steps` updates.

    `labeled` and `unlabeled` index `cases`. Only labeled masks are read.
    For matched replays, pass the same start, seed, training steps, and pool order.
    The same labeled and unlabeled batch sizes are used whenever both pools exist.
    """
    labeled = tuple(labeled)
    unlabeled = tuple(unlabeled)
    if not labeled or len(set(labeled + unlabeled)) != len(labeled) + len(unlabeled):
        raise ValueError("Training requires a nonempty labeled pool and unique, disjoint case IDs.")
    if steps < 0 or batch_size < 1:
        raise ValueError("steps must be nonnegative and batch_size positive.")
    device = configure(device)
    started = perf_counter()
    labeled_pool = SlicePool(cases, labeled)
    unlabeled_pool = SlicePool(cases, unlabeled)
    labeled_rng, unlabeled_rng = [
        np.random.default_rng(sequence) for sequence in np.random.SeedSequence(seed).spawn(2)
    ]
    strong_rng = torch.Generator(device=device).manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        student = UNet().to(device)
    teacher = deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    prior = torch.full((NUM_CLASSES,), 1.0 / NUM_CLASSES, device=device)
    if start is not None:
        student.load_state_dict(start.student)
        teacher.load_state_dict(start.teacher)
        optimizer.load_state_dict(deepcopy(start.optimizer))
        prior.copy_(start.prior)

    stats = torch.zeros(3, device=device)
    for _ in range(steps):
        images, masks = labeled_pool.sample(labeled_rng, batch_size, with_masks=True)
        if labeled_rng.random() < 0.5:
            images = images[:, :, ::-1].copy()
            masks = masks[:, :, ::-1].copy()
        labeled_images = _image_tensor(images, device)
        labeled_masks = torch.from_numpy(masks).to(device, dtype=torch.long)
        student_inputs = labeled_images
        if unlabeled:
            raw, _ = unlabeled_pool.sample(unlabeled_rng, batch_size, with_masks=False)
            weak = _image_tensor(raw, device)
            gain = torch.rand((batch_size, 1, 1, 1), device=device, generator=strong_rng)
            noise = torch.randn(weak.shape, device=device, generator=strong_rng)
            strong = weak * (0.75 + 0.5 * gain) + 0.10 * noise
            student_inputs = torch.cat((labeled_images, strong), dim=0)
            with torch.no_grad():
                with _autocast(device):
                    teacher_logits = teacher(weak)
                targets = teacher_logits.float().softmax(dim=1)
                if uot:
                    targets = _diffunder_uot(targets, prior, labeled_masks)
                valid = targets.amax(dim=1) >= PSEUDO_THRESHOLD
                targets = targets.clamp_min(1e-8).square()
                targets = targets / targets.sum(dim=1, keepdim=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            logits = student(student_inputs)
        logits = logits.float()
        labeled_loss = _supervised_loss(logits[:batch_size], labeled_masks)
        unlabeled_loss = logits.new_zeros(())
        coverage = logits.new_zeros(())
        if unlabeled:
            pixel_loss = -(targets * logits[batch_size:].log_softmax(dim=1)).sum(dim=1)
            unlabeled_loss = (pixel_loss * valid).sum() / valid.sum().clamp_min(1)
            coverage = valid.float().mean()
        (labeled_loss + unlabeled_loss).backward()
        optimizer.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(teacher.parameters()), list(student.parameters()), 1 - EMA_MOMENTUM)
            stats += torch.stack((labeled_loss.detach(), unlabeled_loss.detach(), coverage))

    stats = (stats / steps).cpu().tolist() if steps else (None, None, None)
    return TrainState(
        student=_model_state(student),
        teacher=_model_state(teacher),
        optimizer=_optimizer_state(optimizer),
        prior=prior.detach().cpu().clone(),
        steps=steps + (start.steps if start is not None else 0),
        seconds=perf_counter() - started,
        supervised_loss=stats[0],
        unlabeled_loss=stats[1],
        pseudo_fraction=stats[2],
    )


def predict_many(state, cases, device, *, batch_size=16):
    """Yield one N×4×H×W float32 array per case using the same EMA model."""
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive.")
    device = configure(device)
    with torch.random.fork_rng(devices=[]):
        model = UNet().to(device)
    model.load_state_dict(state.teacher)
    model.eval()
    for case in cases:
        with torch.inference_mode():
            output = np.empty((len(case.images), NUM_CLASSES, *case.images.shape[-2:]), dtype=np.float32)
            for start in range(0, len(case.images), batch_size):
                stop = min(start + batch_size, len(case.images))
                images = _image_tensor(case.images[start:stop], device)
                with _autocast(device):
                    logits = model(images)
                output[start:stop] = logits.float().softmax(dim=1).cpu().numpy()
        yield output


def predict(state, case, device):
    """Return the EMA teacher's float32 class probabilities for one case."""
    return next(predict_many(state, (case,), device))
