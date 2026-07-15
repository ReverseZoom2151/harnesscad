"""The NUMERIC surface -- the deterministic building blocks of generative models.

``domain/numeric`` carries the classical, stdlib-only halves of a pile of
generative-model papers: discrete/categorical diffusion transition algebra, the
clean flow-ODE endpoint, sqrt / integral / augmented noise schedules, the
multiscale attention + pyramid primitives, the distillation gradients, and the
state-space (ZOH) discretisation with its directional scans. The trained weights
those papers also carry live OUTSIDE the repo (exactly like the drawings
subsystem): what is committed here is only the deterministic arithmetic, and
none of it was reachable. This module is the dispatcher.

    transition(num_classes, beta)        -> a categorical forward-corruption matrix
    mask_replace(num_classes, ...)       -> the mask-and-replace transition matrix
    joint_posterior(...)                 -> the DDPM posterior mean on a joint vector
    augmented_schedule(keep, ...)        -> the variance-augmented alpha schedule
    flow_endpoint(eps, mu, s)            -> the clean flow-ODE sample endpoint
    noise_schedule(steps, t)             -> beta / alpha_bar / snr at a step
    integral_transport(seed, coverage)   -> reference-noise transported to cells
    windowed_attention(q, k, v)          -> masked multiscale attention
    pyramid(seq, levels)                 -> a multiscale downsampling pyramid
    distill_gradient(real, fake)         -> the distribution-matching gradient
    decoder_distance(decoded, target)    -> the decoder-consistency distance
    gradient_variance(grads)             -> Adam-scaled gradient variance
    zoh_discretize(a, b, delta)          -> the zero-order-hold discretisation
    directional_scan(...)                -> multi-directional selective scan

Everything here is deterministic and stdlib-only: no trainer runs, no weights are
loaded, nothing shells out. Adapters only -- the numeric modules are never
modified. This is the drawings-subsystem precedent: make the family reachable
without pretending the models are in the repo.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "NumericError",
    "transition",
    "mask_replace",
    "joint_posterior",
    "augmented_schedule",
    "flow_endpoint",
    "noise_schedule",
    "integral_transport",
    "windowed_attention",
    "pyramid",
    "distill_gradient",
    "decoder_distance",
    "gradient_variance",
    "zoh_discretize",
    "directional_scan",
    "discover",
    "groups",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_NUM = "harnesscad.domain.numeric."


class NumericError(ValueError):
    """Base class for every numeric-surface failure."""


# --------------------------------------------------------------------------- #
# Diffusion: discrete / categorical transition algebra
# --------------------------------------------------------------------------- #
def transition(num_classes: int, beta: float) -> List[List[float]]:
    """A uniform categorical forward-corruption matrix (D3PM-style)."""
    from harnesscad.domain.numeric.categorical_diffusion import uniform_transition_matrix

    return uniform_transition_matrix(int(num_classes), float(beta))


def mask_replace(num_classes: int, alpha: float, gamma: float) -> List[List[float]]:
    """The mask-and-replace transition matrix (VQ-Diffusion), one step."""
    from harnesscad.domain.numeric.mask_and_replace import mask_and_replace_matrix

    return mask_and_replace_matrix(int(num_classes), float(alpha), float(gamma))


def joint_posterior(vec_t: Sequence[float], vec_0: Sequence[float], alpha_t: float,
                    alpha_bar_t: float, alpha_bar_prev: float) -> List[float]:
    """The DDPM posterior mean over a joint discrete+continuous state vector."""
    from harnesscad.domain.numeric.joint_diffusion import joint_posterior_mean

    return joint_posterior_mean([float(v) for v in vec_t], [float(v) for v in vec_0],
                                float(alpha_t), float(alpha_bar_t), float(alpha_bar_prev))


def augmented_schedule(discrete_keep: Sequence[float], num_classes: int,
                       k: float = 0.99) -> List[float]:
    """The variance-augmented alpha schedule for continuous relaxation."""
    from harnesscad.domain.numeric.variance_augmentation import augmented_schedule as _augmented

    return _augmented([float(v) for v in discrete_keep], int(num_classes), float(k))


# --------------------------------------------------------------------------- #
# Flow / ODE
# --------------------------------------------------------------------------- #
def flow_endpoint(eps_tilde: float, mu: float, s: float, steps: int = 200,
                  t_start: float = 0.999, t_end: float = 0.001) -> float:
    """Integrate the clean flow-ODE (cosine schedule) to its sample endpoint."""
    from harnesscad.domain.numeric.flow_ode import clean_flow_ode_endpoint

    return clean_flow_ode_endpoint(float(eps_tilde), float(mu), float(s),
                                   steps=int(steps), t_start=float(t_start),
                                   t_end=float(t_end))


# --------------------------------------------------------------------------- #
# Noise schedules
# --------------------------------------------------------------------------- #
def noise_schedule(steps: int, t: int, offset: float = 1e-4) -> Dict[str, float]:
    """The sqrt noise schedule sampled at one step: beta / alpha / alpha_bar / snr."""
    from harnesscad.domain.numeric.sqrt_noise_schedule import SqrtNoiseSchedule

    sched = SqrtNoiseSchedule(steps=int(steps), offset=float(offset))
    ti = int(t)
    return {"t": ti, "beta": sched.beta(ti), "alpha": sched.alpha(ti),
            "alpha_bar": sched.alpha_bar(ti), "snr": sched.snr(ti)}


def integral_transport(seed: int, coverage: Sequence[Sequence[int]]) -> List[float]:
    """Transport a reference-noise field onto a set of coverage cells."""
    from harnesscad.domain.numeric.integral_noise import ReferenceNoise, transport

    ref = ReferenceNoise(seed=int(seed))
    return transport(ref, [[int(c) for c in cells] for cells in coverage])


# --------------------------------------------------------------------------- #
# Multiscale
# --------------------------------------------------------------------------- #
def windowed_attention(q: Sequence[Sequence[float]], k: Sequence[Sequence[float]],
                       v: Sequence[Sequence[float]],
                       mask: Optional[Sequence[Sequence[float]]] = None) -> List[List[float]]:
    """Masked multiscale (windowed) attention over query / key / value rows."""
    from harnesscad.domain.numeric.multiscale_attention import masked_attention

    return masked_attention(q, k, v, mask=mask)


def pyramid(seq: Sequence[Sequence[float]], levels: int, factor: int = 2,
            mode: str = "avg") -> Tuple[Any, ...]:
    """A multiscale downsampling pyramid of a sequence (coarse-to-fine levels)."""
    from harnesscad.domain.numeric.multiscale_pyramid import build_pyramid

    return build_pyramid(seq, int(levels), factor=int(factor), mode=str(mode))


# --------------------------------------------------------------------------- #
# Distillation
# --------------------------------------------------------------------------- #
def distill_gradient(score_real: Sequence[float], score_fake: Sequence[float],
                     weight: float = 1.0) -> List[float]:
    """The distribution-matching distillation gradient (DMD)."""
    from harnesscad.domain.numeric.dmd_distillation import dmd_gradient

    return dmd_gradient([float(v) for v in score_real],
                        [float(v) for v in score_fake], float(weight))


def decoder_distance(decoded: Sequence[float], target: Sequence[float]) -> float:
    """The decoder-consistency distance between a decoded sample and its target."""
    from harnesscad.domain.numeric.decoder_regularizer import decoder_distance as _distance

    return _distance([float(v) for v in decoded], [float(v) for v in target])


def gradient_variance(grads: Sequence[Sequence[float]], beta1: float = 0.9,
                      beta2: float = 0.999, bias_correction: bool = True,
                      eps: float = 1e-12) -> float:
    """The Adam-scaled variance of a batch of gradients."""
    from harnesscad.domain.numeric.gradient_variance import scaled_gradient_variance

    return scaled_gradient_variance(grads, beta1=float(beta1), beta2=float(beta2),
                                    bias_correction=bool(bias_correction), eps=float(eps))


# --------------------------------------------------------------------------- #
# State-space (selective scan)
# --------------------------------------------------------------------------- #
def zoh_discretize(a_diag: Sequence[float], b_diag: Sequence[float], delta: float,
                   simplified: bool = False) -> Tuple[List[float], List[float]]:
    """The zero-order-hold discretisation of a diagonal state-space model."""
    from harnesscad.domain.numeric.zoh_discretization import discretize

    return discretize([float(v) for v in a_diag], [float(v) for v in b_diag],
                      float(delta), simplified=bool(simplified))


def directional_scan(z_seq: Sequence[Sequence[float]], a_seq: Sequence[Sequence[float]],
                     b_seq: Sequence[Sequence[float]], c_seq: Sequence[Sequence[float]],
                     g_seq: Sequence[Sequence[float]],
                     orders: Sequence[Sequence[int]], mode: str = "sum",
                     h0: Optional[Sequence[float]] = None) -> List[Any]:
    """A multi-directional selective scan, merged across scan orders."""
    from harnesscad.domain.numeric.scan_directions import multidirectional_scan

    return multidirectional_scan(z_seq, a_seq, b_seq, c_seq, g_seq,
                                 tuple(tuple(int(i) for i in o) for o in orders),
                                 mode=str(mode), h0=h0)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e
            for e in capability_registry.find(package="numeric")}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("diffusion", "transition", _NUM + "categorical_diffusion",
     "categorical (D3PM) forward-corruption transition matrices + posterior"),
    ("diffusion", "mask_replace", _NUM + "mask_and_replace",
     "the mask-and-replace transition (VQ-Diffusion) and its cumulative params"),
    ("diffusion", "joint_posterior", _NUM + "joint_diffusion",
     "the DDPM posterior over a joint discrete+continuous sketch state"),
    ("diffusion", "augmented_schedule", _NUM + "variance_augmentation",
     "the variance-augmented (Gumbel) continuous relaxation of a discrete schedule"),
    ("flow", "flow_endpoint", _NUM + "flow_ode",
     "the clean flow-ODE endpoint + EDM Heun sampler (cosine schedule)"),
    ("schedule", "noise_schedule", _NUM + "sqrt_noise_schedule",
     "the sqrt noise schedule: beta / alpha_bar / snr, forward diffuse, quantise"),
    ("schedule", "integral_transport", _NUM + "integral_noise",
     "integral / reference-noise transport onto coverage cells"),
    ("multiscale", "windowed_attention", _NUM + "multiscale_attention",
     "masked windowed attention, adaptive fusion, sequence-aware position"),
    ("multiscale", "pyramid", _NUM + "multiscale_pyramid",
     "multiscale + Laplacian pyramids (downsample / upsample / reconstruct)"),
    ("distill", "distill_gradient", _NUM + "dmd_distillation",
     "distribution-matching distillation gradients + few-step timesteps"),
    ("distill", "decoder_distance", _NUM + "decoder_regularizer",
     "the decoder-consistency regularisation energy over a diffusion trajectory"),
    ("distill", "gradient_variance", _NUM + "gradient_variance",
     "Adam-scaled gradient variance from EMA moments"),
    ("statespace", "zoh_discretize", _NUM + "zoh_discretization",
     "zero-order-hold discretisation of a diagonal SSM + discrete scan"),
    ("statespace", "directional_scan", _NUM + "scan_directions",
     "multi-directional selective scans merged across scan orders"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def groups() -> Tuple[str, ...]:
    return tuple(sorted({g for g, _n, _m, _d in _ROUTES}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, "no route yet") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every numeric route (default)")
    parser.add_argument("--groups", action="store_true",
                        help="list the numeric theme groups")
    parser.add_argument("--schedule", default=None, metavar="STEPS,T",
                        help="sample the sqrt noise schedule at step T of STEPS")
    parser.add_argument("--transition", default=None, metavar="NUM_CLASSES,BETA",
                        help="a uniform categorical transition matrix")
    parser.add_argument("--unadapted", action="store_true",
                        help="list numeric modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "groups", False):
        for g in groups():
            print(g)
        return 0

    if getattr(args, "schedule", None):
        steps_s, t_s = args.schedule.split(",")
        row = noise_schedule(int(steps_s), int(t_s))
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0

    if getattr(args, "transition", None):
        nc_s, beta_s = args.transition.split(",")
        mat = transition(int(nc_s), float(beta_s))
        if getattr(args, "json", False):
            print(json.dumps(mat, indent=2))
        else:
            for row in mat:
                print(" ".join("%.4f" % v for v in row))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-11s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad numeric",
        description="numeric surface: the deterministic building blocks of "
                    "generative models (diffusion, flow, schedules, multiscale, "
                    "distillation, state-space)")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
