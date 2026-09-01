"""
src/augment.py

Train-time augmentation for batched landmark tensors of shape
(B, T, L, 3), where L is the active landmark set (all 50, or hands only).

Nothing here touches src/preprocessing.py or the stored arrays: these are
transforms applied to a batch on its way into the model, so evaluation
data can be left alone simply by not calling them.

Three transforms, applied in this order:

1. mirror  -- reflect the signer left/right. Both a coordinate flip and a
              landmark reindex; see mirror_permutation.
2. warp    -- resample the time axis through a random monotone
              reparametrization, i.e. perform the sign a little faster or
              slower, unevenly.
3. jitter  -- additive Gaussian noise on coordinates, standing in for
              landmark-detection error.

**Coordinate frame.** mirror() assumes shoulder-centered coordinates, where
the reflection is x -> -x. On un-normalized input the body midline is not
at x = 0 (raw x lies in roughly [0, 1]) and this reflection is wrong. The
caller is responsible for not combining the two.

**Landmarks that are exactly zero mean "this hand is not used for this
sign"**, which is a real signal rather than a coordinate. warp preserves it
for free (interpolating zeros gives zeros) and mirror moves it to the other
hand's slot, which is the intended behaviour. jitter would destroy it by
turning an absent hand into a noisy present one, so it masks those
landmarks out explicitly.
"""
from __future__ import annotations

import torch

import preprocessing as pp

# Left/right landmark pairs, as positions in the full 50-landmark ordering.
# Hands occupy 0-20 and 21-41; the eight pose points are stored as
# alternating left/right (shoulders, elbows, wrists, hips), so each
# consecutive even/odd pair is a mirror pair.
NUM_HAND_LANDMARKS = len(pp.HAND_INDICES)
NUM_POSE_LANDMARKS = len(pp.POSE_INDICES)
HANDS_END = 2 * NUM_HAND_LANDMARKS


def mirror_permutation(num_landmarks: int, device: torch.device) -> torch.Tensor:
    """Index permutation that swaps left-side and right-side landmarks.

    Reflecting the coordinates alone is not enough: after a flip, what was
    the left hand is physically on the right, so the two hand blocks have
    to trade places as well, along with each left/right pose pair. Applying
    only one half of this produces anatomically impossible inputs.

    Works for either active landmark set -- the full 50, or the 42 hand
    landmarks alone."""
    n = NUM_HAND_LANDMARKS
    order = list(range(num_landmarks))
    order[0:n] = list(range(n, 2 * n))
    order[n : 2 * n] = list(range(0, n))
    for i in range(HANDS_END, num_landmarks - 1, 2):
        order[i], order[i + 1] = order[i + 1], order[i]
    return torch.tensor(order, dtype=torch.long, device=device)


def mirror(x: torch.Tensor, perm: torch.Tensor, p: float) -> torch.Tensor:
    """Reflect a random subset of the batch about the body midline."""
    if p <= 0.0:
        return x
    b = x.shape[0]
    flip = torch.rand(b, device=x.device) < p
    if not bool(flip.any()):
        return x
    out = x.clone()
    sub = out[flip][:, :, perm, :]
    sub[..., 0] = -sub[..., 0]
    out[flip] = sub
    return out


def warp(x: torch.Tensor, max_warp: float) -> torch.Tensor:
    """Resample the time axis through a random monotone reparametrization.

    Frame boundaries are preserved (the warp maps 0 -> 0 and T-1 -> T-1),
    so this changes the pacing within a sign rather than trimming it."""
    if max_warp <= 0.0:
        return x
    b, t, l, c = x.shape
    steps = 1.0 + (torch.rand(b, t - 1, device=x.device) * 2.0 - 1.0) * max_warp
    cumulative = torch.cumsum(steps, dim=1)
    cumulative = torch.cat([torch.zeros(b, 1, device=x.device), cumulative], dim=1)
    src = cumulative / cumulative[:, -1:] * (t - 1)  # monotone, 0 .. t-1

    lo = src.floor().long().clamp(0, t - 1)
    hi = (lo + 1).clamp(max=t - 1)
    frac = (src - lo.to(src.dtype)).view(b, t, 1, 1)
    index_lo = lo.view(b, t, 1, 1).expand(b, t, l, c)
    index_hi = hi.view(b, t, 1, 1).expand(b, t, l, c)
    return x.gather(1, index_lo) * (1.0 - frac) + x.gather(1, index_hi) * frac


def jitter(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive Gaussian noise, skipping landmarks that are zero for the
    whole sequence. Those zeros encode "hand not used for this sign"; noise
    on them would turn an absent hand into a spurious present one."""
    if sigma <= 0.0:
        return x
    active = (x.abs().sum(dim=(1, 3), keepdim=True) > 0).to(x.dtype)
    return x + torch.randn_like(x) * sigma * active


def augment_batch(
    x: torch.Tensor,
    perm: torch.Tensor,
    p_mirror: float = 0.5,
    max_warp: float = 0.2,
    jitter_sigma: float = 0.01,
) -> torch.Tensor:
    """Apply all three transforms to a (B, T, L, 3) batch. Jitter runs last
    so the noise it adds isn't smoothed away by the time warp."""
    x = mirror(x, perm, p_mirror)
    x = warp(x, max_warp)
    x = jitter(x, jitter_sigma)
    return x


def shuffle_time(x: torch.Tensor) -> torch.Tensor:
    """Independently permute the frames of every sequence in the batch.

    This is a diagnostic rather than an augmentation: a model trained and
    evaluated on shuffled frames cannot use temporal order at all, so its
    accuracy measures how much of the task is solvable from the unordered
    collection of poses a sequence contains."""
    b, t = x.shape[0], x.shape[1]
    order = torch.argsort(torch.rand(b, t, device=x.device), dim=1)
    index = order.view(b, t, 1, 1).expand_as(x)
    return x.gather(1, index)
