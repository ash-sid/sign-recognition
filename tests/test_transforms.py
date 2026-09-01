"""
tests/test_transforms.py

Checks the preprocessing and augmentation transforms against synthetic
data, so their behaviour can be verified without the full dataset present.

The properties asserted here are the ones that are easy to get subtly
wrong and impossible to notice from a training curve: that mirroring
reindexes landmarks as well as flipping coordinates, that transforms
preserve the exact zeros encoding "this hand is not used for this sign",
and that two scripts writing to the same report file leave each other's
sections alone.

Run: uv run python tests\\test_transforms.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import augment  # noqa: E402
import preprocessing as pp  # noqa: E402
import report  # noqa: E402

ROOT = Path(__file__).parent
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


# --- synthetic parquet-shaped sequence --------------------------------------

def make_sequence(n_frames=30, left_present=True, right_present=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for t, idxs in (("left_hand", range(21)), ("right_hand", range(21)),
                        ("pose", range(33)), ("face", range(468))):
            for i in idxs:
                if t == "left_hand" and not left_present:
                    x = y = z = np.nan
                elif t == "right_hand" and not right_present:
                    x = y = z = np.nan
                elif t == "pose" and i == 11:
                    x, y, z = 0.4, 0.3, 0.0          # left shoulder
                elif t == "pose" and i == 12:
                    x, y, z = 0.6, 0.3, 0.0          # right shoulder
                else:
                    x, y, z = rng.uniform(0.2, 0.8, 3)
                rows.append({"frame": f, "row_id": f"{f}-{t}-{i}", "type": t,
                             "landmark_index": i, "x": x, "y": y, "z": z})
    return pd.DataFrame(rows)


# --- 1. preprocessing normalize flag ----------------------------------------

df = make_sequence()
norm = pp.process_sequence(df)
raw = pp.process_sequence(df, normalize=False)
check("normalized output has contract shape", norm.shape == (70, 50, 3), str(norm.shape))
check("un-normalized output has contract shape", raw.shape == (70, 50, 3))
check("normalize=False actually changes the output", not np.allclose(norm, raw))
check("normalize=True is unchanged from the default path",
      np.allclose(norm, pp.process_sequence(df, 70, True)))
check("no NaNs in either variant", not np.isnan(norm).any() and not np.isnan(raw).any())
# shoulders are the normalization reference: after it they sit at fixed positions
check("normalized shoulders are centred", abs(norm[:, 42:44, 0].mean()) < 1e-5,
      f"mean x={norm[:, 42:44, 0].mean():.2e}")
check("raw shoulders keep original coords", abs(raw[0, 42, 0] - 0.4) < 1e-4)

df_one = make_sequence(left_present=False)
one = pp.process_sequence(df_one)
check("unused hand zeroed with normalize=True", np.all(one[:, 0:21, :] == 0.0))
one_raw = pp.process_sequence(df_one, normalize=False)
check("unused hand zeroed with normalize=False", np.all(one_raw[:, 0:21, :] == 0.0))

# --- 2. mirror permutation ---------------------------------------------------

dev = torch.device("cpu")
perm50 = augment.mirror_permutation(50, dev)
check("perm swaps hand blocks",
      perm50[:21].tolist() == list(range(21, 42)) and perm50[21:42].tolist() == list(range(21)))
check("perm swaps pose pairs", perm50[42:].tolist() == [43, 42, 45, 44, 47, 46, 49, 48])
perm42 = augment.mirror_permutation(42, dev)
check("perm works for hands-only set", perm42.tolist() == list(range(21, 42)) + list(range(21)))
check("perm is a valid permutation", sorted(perm50.tolist()) == list(range(50)))

x = torch.randn(8, 70, 50, 3)
once = augment.mirror(x, perm50, p=1.0)
twice = augment.mirror(once, perm50, p=1.0)
check("mirror twice is identity", torch.allclose(x, twice, atol=1e-6))
check("mirror negates x", torch.allclose(once[:, :, 21:42, 0], -x[:, :, 0:21, 0], atol=1e-6))
check("mirror leaves y and z alone", torch.allclose(once[:, :, 21:42, 1], x[:, :, 0:21, 1], atol=1e-6))
check("mirror with p=0 is a no-op", torch.allclose(augment.mirror(x, perm50, 0.0), x))

# an unused (all-zero) left hand should end up as an unused right hand
z = torch.randn(4, 70, 50, 3)
z[:, :, 0:21, :] = 0.0
zm = augment.mirror(z, perm50, p=1.0)
check("mirror moves the absent hand to the other slot",
      bool((zm[:, :, 21:42, :] == 0).all()) and bool((zm[:, :, 0:21, :] != 0).any()))

# --- 3. jitter ---------------------------------------------------------------

j = augment.jitter(z, sigma=0.05)
check("jitter leaves the absent hand exactly zero", bool((j[:, :, 0:21, :] == 0).all()))
check("jitter perturbs present landmarks", not torch.allclose(j[:, :, 21:42, :], z[:, :, 21:42, :]))
check("jitter with sigma=0 is a no-op", torch.allclose(augment.jitter(z, 0.0), z))

# --- 4. time warp ------------------------------------------------------------

w = augment.warp(x, max_warp=0.3)
check("warp preserves shape", w.shape == x.shape)
check("warp preserves the first frame", torch.allclose(w[:, 0], x[:, 0], atol=1e-5))
check("warp preserves the last frame", torch.allclose(w[:, -1], x[:, -1], atol=1e-5))
check("warp changes the interior", not torch.allclose(w[:, 1:-1], x[:, 1:-1], atol=1e-4))
check("warp keeps absent hands at zero", bool((augment.warp(z, 0.3)[:, :, 0:21, :] == 0).all()))
check("warp with max_warp=0 is a no-op", torch.allclose(augment.warp(x, 0.0), x))

# --- 5. shuffle_time ---------------------------------------------------------

s = augment.shuffle_time(x)
check("shuffle preserves shape", s.shape == x.shape)
check("shuffle preserves the frame multiset",
      torch.allclose(s.sum(dim=1), x.sum(dim=1), atol=1e-4))
check("shuffle actually reorders", not torch.allclose(s, x))

# --- 6. report section splicing ---------------------------------------------

tmp = ROOT / "_tmp_report.md"
if tmp.exists():
    tmp.unlink()
report.update_section(tmp, "baselines", "# Results\n\nbaseline body")
report.update_section(tmp, "runs", "## Models\n\nruns body v1")
first = tmp.read_text(encoding="utf-8")
report.update_section(tmp, "runs", "## Models\n\nruns body v2")
second = tmp.read_text(encoding="utf-8")
check("both sections present", "baseline body" in second and "runs body v2" in second)
check("replacing one section leaves the other intact", "baseline body" in second)
check("replacing does not duplicate", second.count("<!-- RUNS_START -->") == 1)
check("old body is gone", "runs body v1" not in second)
report.update_section(tmp, "baselines", "# Results\n\nbaseline body")
check("re-writing an unchanged section is idempotent",
      tmp.read_text(encoding="utf-8") == second)
tmp.unlink()

print()
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
