"""Deterministic efficiency / latency model for Turbo3D (Hu et al., 2024).

Turbo3D's headline claim is an ultra-fast text-to-3D pipeline ("under one
second", 0.35 s at resolution 256). Its speed comes from three deterministic
levers, each of which is a simple, reproducible calculation -- independent of the
learned networks -- that this module models:

  * **Step reduction.** The many-step MV teacher is distilled to a 4-step
    student, giving a denoiser-evaluation speedup equal to the step ratio
    (the paper reports ~50x faster than the teacher).

  * **Latent GS-LRM: decode elimination.** Feeding the reconstructor multi-view
    *latents* instead of pixels skips VAE decoding entirely. Total pipeline
    latency drops from ``generate + decode + reconstruct`` to
    ``generate + reconstruct``. The paper: 0.45 s -> 0.35 s at res 256 (~22%),
    and 1.62 s -> 1.28 s at res 512 (~21%).

  * **Latent GS-LRM: sequence halving.** A latent is downsampled vs the pixel
    image (an ``f``-fold VAE downsample per axis), so the transformer token count
    per view shrinks by ``f^2``. The paper notes this "halves the transformer
    sequence length" for their configuration.

This module packages those as pure functions plus a small ``PipelineLatency``
dataclass, and a helper to reproduce the reported speedup percentages. No wall
clock, no randomness -- every number is a deterministic function of the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# --------------------------------------------------------------------------- #
# Step-reduction speedup
# --------------------------------------------------------------------------- #
def step_speedup(teacher_steps: int, student_steps: int) -> float:
    """Denoiser-evaluation speedup ``teacher_steps / student_steps``.

    The diffusion cost is dominated by the number of denoiser passes, so the step
    ratio is the leading speedup of distillation (Turbo3D: ~50x vs the teacher).
    """
    if teacher_steps <= 0 or student_steps <= 0:
        raise ValueError("step counts must be positive")
    return teacher_steps / student_steps


# --------------------------------------------------------------------------- #
# Transformer sequence length (tokens) per view
# --------------------------------------------------------------------------- #
def token_count(resolution: int, patch_size: int, downsample: int = 1) -> int:
    """Transformer tokens for one ``resolution`` x ``resolution`` view.

    A pixel-space model tokenises the image at ``patch_size`` (``downsample=1``).
    A latent-space model first applies an ``downsample``-fold VAE downsample per
    axis, so the token grid is ``(resolution / downsample / patch_size)^2``. The
    resolution must divide evenly by ``downsample * patch_size``.
    """
    if resolution <= 0 or patch_size <= 0 or downsample <= 0:
        raise ValueError("resolution, patch_size, downsample must be positive")
    eff = resolution // downsample
    if eff % patch_size != 0 or resolution % downsample != 0:
        raise ValueError("resolution must be divisible by downsample*patch_size")
    grid = eff // patch_size
    return grid * grid


def sequence_length_ratio(resolution: int, patch_size: int, downsample: int) -> float:
    """Pixel-token / latent-token ratio, i.e. how much the sequence shrinks.

    Equals ``downsample^2`` (token count scales with area). For the paper's
    configuration this is the factor by which the latent GS-LRM shortens the
    transformer sequence versus a pixel GS-LRM.
    """
    pixel = token_count(resolution, patch_size, downsample=1)
    latent = token_count(resolution, patch_size, downsample=downsample)
    return pixel / latent


# --------------------------------------------------------------------------- #
# Whole-pipeline latency
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PipelineLatency:
    """Latency breakdown (seconds) of a text-to-3D pipeline.

    ``generate`` is the MV generator time, ``decode`` the VAE decode time (only
    incurred by a pixel-space reconstructor), and ``reconstruct`` the GS-LRM time.
    """

    generate: float
    decode: float
    reconstruct: float

    def total(self, latent: bool) -> float:
        """Total latency; a ``latent`` reconstructor skips the decode stage."""
        if self.generate < 0 or self.decode < 0 or self.reconstruct < 0:
            raise ValueError("latency components must be non-negative")
        if latent:
            return self.generate + self.reconstruct
        return self.generate + self.decode + self.reconstruct


def latent_speedup_fraction(latency: PipelineLatency) -> float:
    """Fractional latency saved by going latent (skipping decode), in ``[0, 1)``.

    ``(pixel_total - latent_total) / pixel_total = decode / pixel_total``. The
    paper reports ~0.22 at res 256 and ~0.21 at res 512.
    """
    pixel = latency.total(latent=False)
    if pixel <= 0.0:
        raise ValueError("pixel-path total latency must be positive")
    return latency.decode / pixel


def speedup_percent(latency: PipelineLatency) -> float:
    """Latent speedup expressed as a percentage (0-100)."""
    return 100.0 * latent_speedup_fraction(latency)


# --------------------------------------------------------------------------- #
# Convenience: reproduce the paper's headline table figures
# --------------------------------------------------------------------------- #
def under_one_second(total_latency: float) -> bool:
    """Whether the pipeline meets Turbo3D's "under one second" claim."""
    return total_latency < 1.0


def compare_methods_by_time(latencies: dict) -> List[str]:
    """Method names sorted fastest-first by inference latency (stable on ties).

    ``latencies`` maps method name -> seconds (e.g. Turbo3D 0.35, TripoSR 1.19,
    LGM 6.56, ...). Returns the names ordered ascending, so the fastest method
    (Turbo3D in Table 1) is first.
    """
    for v in latencies.values():
        if v < 0:
            raise ValueError("latencies must be non-negative")
    return sorted(latencies, key=lambda k: latencies[k])
