"""VoxHammer feature/latent preservation via masked blending.

Training-free feature replacement: during denoising the preserved-region latents
and attention key/value tokens are overwritten by their cached inverted
counterparts, while the edited region is denoised freely. This module implements
the deterministic replacement operators of this approach:

- structure-stage latent blend:
      z <- M (x) z_new + (1 - M) (x) z_hat
- sparse-latent hard replacement:
      for u in Omega_keep:  z[u] <- z_hat[u]
- attention key/value replacement (eqs. 6-7):
      K <- W (x) K_new + (1 - W) (x) K_cache
      V <- W (x) V_new + (1 - W) (x) V_cache
- optional attention masks blocking mixing between edited and preserved tokens.

Latents and K/V tokens are represented as tuples of floats keyed by voxel/token
coordinate. All operations are deterministic and stdlib-only.
"""
from __future__ import annotations


def blend_vectors(new, preserved, w):
    """Elementwise convex blend ``w*new + (1-w)*preserved`` for one vector."""
    new = tuple(float(x) for x in new)
    preserved = tuple(float(x) for x in preserved)
    if len(new) != len(preserved):
        raise ValueError("new and preserved must have equal length")
    w = float(w)
    return tuple(w * n + (1.0 - w) * p for n, p in zip(new, preserved))


def blend_latents(new_map, preserved_map, mask):
    """Per-voxel masked blend.

    For every coordinate in ``new_map`` the weight ``mask[coord]`` (default 0.0,
    i.e. fully preserved) selects between the freshly denoised latent and the
    cached inverted latent. Coordinates absent from ``preserved_map`` keep the
    new latent unchanged (nothing to preserve there).
    """
    out = {}
    for coord, new_vec in new_map.items():
        w = float(mask.get(coord, 0.0))
        if coord in preserved_map:
            out[coord] = blend_vectors(new_vec, preserved_map[coord], w)
        else:
            out[coord] = tuple(float(x) for x in new_vec)
    return out


def hard_replace(latents, cached, keep_set):
    """Hard preserved-region replacement.

    Returns a copy of ``latents`` with every coordinate in ``keep_set``
    overwritten by its cached inverted latent.
    """
    keep = frozenset(keep_set)
    out = {}
    for coord, vec in latents.items():
        if coord in keep and coord in cached:
            out[coord] = tuple(float(x) for x in cached[coord])
        else:
            out[coord] = tuple(float(x) for x in vec)
    return out


def kv_replace(new, cache, w):
    """Attention K (or V) token replacement for a single token (eqs. 6-7)."""
    return blend_vectors(new, cache, w)


def masked_kv_replace(new_map, cache_map, w_map):
    """Masked K/V replacement over a token map (eqs. 6-7).

    ``w_map[token]`` is the edit weight W (1 keeps the new token, 0 restores the
    cached token). Tokens without a cached counterpart are left as-is.
    """
    out = {}
    for token, new_vec in new_map.items():
        w = float(w_map.get(token, 0.0))
        if token in cache_map:
            out[token] = blend_vectors(new_vec, cache_map[token], w)
        else:
            out[token] = tuple(float(x) for x in new_vec)
    return out


def attention_allow_mask(is_edit):
    """Block cross-group attention between edited and preserved tokens.

    ``is_edit`` is an ordered sequence of booleans (one per token). Returns a
    2D boolean matrix ``A`` where ``A[i][j]`` is True iff token j may attend to
    token i, i.e. only tokens of the same group (both edited or both preserved)
    are allowed to mix. This is the optional attention mask that prevents the
    edit concept from leaking into preserved tokens.
    """
    flags = [bool(x) for x in is_edit]
    n = len(flags)
    return [[flags[i] == flags[j] for j in range(n)] for i in range(n)]
