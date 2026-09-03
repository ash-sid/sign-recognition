"""
src/error_analysis.py

Reads the per-sequence predictions written by src/evaluate.py, joins them to
the per-sequence metadata written by src/cache_dataset.py, and asks which
kinds of sequence a model gets wrong.

Headline accuracy is one number over a heterogeneous test set. This script
breaks it apart along the properties that were recorded precisely because
the fixed-shape model input throws them away: who signed a sequence, which
hand they used, how well each hand was tracked, and how long the sequence
was before resampling.

Four analyses and a caveat:

  Hand slot     Which of the two hand blocks a signer's data occupies, per
                signer, and how that differs between splits. Paired against
                a mirrored scoring of the same sequences, which measures
                dependence on the slot directly rather than inferring it
                from the handful of signers a split contains.
  Tracking      Accuracy against how much of the active hand was actually
                detected, since a landmark file with most frames missing is
                a different input from a complete one.
  Length        Accuracy against the pre-resampling frame count, broken
                down per signer so a difference in signing tempo is not
                mistaken for a difference in sign length.
  Collisions    Which pairs of signs are mutually confusable, ranked so
                that common signs do not dominate the list by frequency
                alone.

Per-class accuracy is reported as a distribution rather than a ranking. At
roughly 56 test sequences per sign a single class's accuracy carries a 95%
interval near +/-13pp, so a table of the worst individual signs would
mostly rank sampling noise.

Run: uv run python src\\error_analysis.py --run-name abl_hands_aug --split test
     uv run python src\\error_analysis.py --run-name abl_hands_aug --split test --compare-run abl_hands_only_long
Requires reports/preds_<run>_<split>.csv (src\\evaluate.py) and
data/cache/meta_<split>.csv (src\\cache_dataset.py).
Writes reports/per_class_<run>_<split>.csv, reports/confusion_<run>_<split>.csv,
reports/confusion_<run>_<split>.png, and a section of reports/results.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a training machine; write files only
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import cache_dataset as cache  # noqa: E402
import report  # noqa: E402

REPORTS = Path("reports")
RESULTS_MD = REPORTS / "results.md"
SPLITS = ("train", "val", "test")

# A signer whose one-handed sequences use the right hand this often or more
# is treated as occupying the right-hand slot, and the mirror image for the
# left. The gap between the two thresholds is wide because the interesting
# quantity is which signers are unambiguous, not where exactly to cut.
RIGHT_SLOT_THRESHOLD = 80.0
LEFT_SLOT_THRESHOLD = 20.0

# Pre-resampling frame counts. TARGET_LEN is a boundary because it is where
# resampling stops padding a sequence out and starts discarding frames from
# it; the median and p95 come from the full-dataset length distribution.
LENGTH_BINS = [0, 22, 70, 135, 10**9]
LENGTH_LABELS = ["<=22", "23-70", "71-135", ">135"]

# Fraction of frames in which the better-tracked hand was not detected.
TRACKING_BINS = [-0.001, 0.1, 0.25, 0.5, 1.0]
TRACKING_LABELS = ["<10%", "10-25%", "25-50%", ">50%"]

TOP_COLLISIONS = 15
CONFUSION_FIGURE_CLASSES = 40


# --- Loading ----------------------------------------------------------------

def preds_path(run_name: str, split: str, mirrored: bool = False) -> Path:
    suffix = "_mirrored" if mirrored else ""
    return REPORTS / f"preds_{run_name}_{split}{suffix}.csv"


def load_joined(run_name: str, split: str) -> pd.DataFrame:
    """Predictions joined to metadata, one row per scored sequence.

    Both files are written in the same filtered dataset order, so they join
    positionally. That is checked rather than assumed: a silent
    misalignment here would attribute every prediction to the wrong signer
    and the resulting analysis would look entirely reasonable."""
    p_path = preds_path(run_name, split)
    m_path = cache.meta_path(split)
    for path, builder in ((p_path, "src\\evaluate.py"), (m_path, "src\\cache_dataset.py")):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Build it first with {builder}.")

    preds = pd.read_csv(p_path)
    meta = pd.read_csv(m_path)
    if len(preds) != len(meta):
        raise SystemExit(
            f"{p_path.name} has {len(preds)} rows and {m_path.name} has {len(meta)}. "
            f"They describe different sets of sequences and cannot be joined."
        )
    disagree = int((preds["true"].to_numpy() != meta["sign"].to_numpy()).sum())
    if disagree:
        raise SystemExit(
            f"{p_path.name} and {m_path.name} disagree on the true sign for {disagree} "
            f"of {len(preds)} rows, so their row orders do not match."
        )

    joined = pd.concat([preds, meta.drop(columns=["sign"])], axis=1)
    joined["correct"] = joined["true_rank"] == 1
    joined["correct5"] = joined["true_rank"] <= 5
    joined["one_handed"] = joined["left_hand_unused"] != joined["right_hand_unused"]
    # The hand that was actually tracked is the better-detected one; for a
    # one-handed sequence the other block is absent by construction and its
    # NaN rate says nothing about tracking quality.
    joined["active_nan_frac"] = joined[["left_hand_nan_frac", "right_hand_nan_frac"]].min(axis=1)
    return joined


def load_all_meta() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = cache.meta_path(split)
        if not path.exists():
            raise SystemExit(f"Missing {path}. Build it with src\\cache_dataset.py --metadata-only.")
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --- Statistics -------------------------------------------------------------

def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """95% confidence interval for a proportion.

    Wilson rather than the textbook normal approximation because the latter
    misbehaves for the small per-class counts and near-0 or near-1 rates
    that appear here."""
    if total == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Compare two scorings of the same sequences.

    The two conditions see identical inputs, so treating their accuracies
    as independent samples would badly overstate the uncertainty. Only the
    sequences the two disagree on carry information; this returns those two
    counts and the two-sided exact binomial p-value over them."""
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    if b + c == 0:
        return b, c, 1.0
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b, c, p


def group_accuracy(df: pd.DataFrame, by: str, order: list[str] | None = None) -> pd.DataFrame:
    """Accuracy per group, in the order the groups were defined.

    order is passed explicitly because these buckets are ranges whose labels
    do not sort meaningfully as text: "<10%" and ">50%" would land either
    side of "25-50%" alphabetically, presenting a monotone trend as noise."""
    rows = []
    for value, sub in df.groupby(by, observed=True):
        correct = int(sub["correct"].sum())
        lo, hi = wilson_interval(correct, len(sub))
        rows.append(
            {
                by: str(value),
                "n": len(sub),
                "top1": correct / len(sub),
                "ci_lo": lo,
                "ci_hi": hi,
                "top5": float(sub["correct5"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    if order is not None and len(out):
        out[by] = pd.Categorical(out[by], categories=order, ordered=True)
        out = out.sort_values(by).reset_index(drop=True)
    return out


# --- Analyses ---------------------------------------------------------------

def signer_hand_slots(meta: pd.DataFrame) -> pd.DataFrame:
    """Which hand block each signer's data occupies.

    Measured over one-handed sequences only: a two-handed sequence fills
    both blocks and says nothing about which one is dominant. Note this
    describes the data, not the person. MediaPipe's left/right assignment
    depends on whether the capture was mirrored, so a left-slot signer and
    a right-handed signer recorded in a flipped frame are indistinguishable
    from landmarks alone. The distinction does not matter to the model,
    which only ever sees which block is filled."""
    one_handed = meta[meta["left_hand_unused"] != meta["right_hand_unused"]]
    grouped = one_handed.groupby(["split", "participant_id"])
    out = grouped["left_hand_unused"].mean().mul(100).rename("pct_right").reset_index()
    out["n_one_handed"] = grouped.size().to_numpy()
    out["slot"] = np.where(
        out["pct_right"] >= RIGHT_SLOT_THRESHOLD,
        "right",
        np.where(out["pct_right"] <= LEFT_SLOT_THRESHOLD, "left", "mixed"),
    )
    return out.sort_values("pct_right").reset_index(drop=True)


def mirror_probe(run_name: str, split: str, joined: pd.DataFrame) -> dict[str, object] | None:
    """Compare a scoring of the split against a scoring of its mirror image.

    Reflecting a sequence is a valid re-performance of the same sign by
    someone whose dominant hand is the other one. A model that has not
    keyed on which block is filled should score the same either way."""
    path = preds_path(run_name, split, mirrored=True)
    if not path.exists():
        return None
    mirrored = pd.read_csv(path)
    if len(mirrored) != len(joined):
        raise SystemExit(f"{path.name} has {len(mirrored)} rows, expected {len(joined)}.")

    m_correct = (mirrored["true_rank"] == 1).to_numpy()
    o_correct = joined["correct"].to_numpy()
    b, c, p = mcnemar(o_correct, m_correct)
    return {
        "top1": float(o_correct.mean()),
        "mirrored_top1": float(m_correct.mean()),
        "delta": float(m_correct.mean() - o_correct.mean()),
        "lost": b,
        "gained": c,
        "p": p,
        # Identical accuracy does not imply identical predictions: a model
        # can be symmetric in aggregate while flipping many individual
        # sequences. This separates the two.
        "churn": float((mirrored["pred_1"].to_numpy() != joined["pred_1"].to_numpy()).mean()),
    }


def collisions(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Sign pairs the model confuses in both directions.

    Ranked by the sum of the two conditional error rates rather than by raw
    counts, so a pair is surfaced for being mutually confusable and not for
    being common. A one-directional confusion scores half as highly as a
    symmetric one of the same size, which is the intended ordering: a pair
    the model cannot separate in either direction is the more interesting
    failure."""
    errors = df[~df["correct"]]
    counts = errors.groupby(["true", "pred_1"], observed=True).size()
    per_class = df.groupby("true", observed=True).size()

    rates: dict[tuple[str, str], float] = {}
    for (true_sign, pred_sign), count in counts.items():
        rates[(true_sign, pred_sign)] = count / per_class[true_sign]

    rows = []
    seen: set[tuple[str, str]] = set()
    for (a, b), rate_ab in rates.items():
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        rate_ba = rates.get((b, a), 0.0)
        rows.append(
            {
                "sign_a": key[0],
                "sign_b": key[1],
                "a_to_b": counts.get((key[0], key[1]), 0),
                "b_to_a": counts.get((key[1], key[0]), 0),
                "symmetric_rate": rate_ab + rate_ba,
                "bidirectional": rate_ab > 0 and rate_ba > 0,
            }
        )
    out = pd.DataFrame(rows).sort_values("symmetric_rate", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def confusion_figure(df: pd.DataFrame, path: Path, n_classes: int) -> None:
    """Confusion matrix over the classes that account for the most error.

    A 250x250 image is decoration -- no cell is legible and no pattern is
    readable. Restricting it to the classes the model actually struggles
    with produces a figure worth looking at; the full matrix is written to
    CSV alongside for anything that needs it."""
    worst = (
        df.groupby("true", observed=True)["correct"].mean().sort_values().head(n_classes).index.tolist()
    )
    sub = df[df["true"].isin(worst)]
    matrix = pd.crosstab(sub["true"], sub["pred_1"]).reindex(index=worst, columns=worst, fill_value=0)
    normalized = matrix.div(matrix.to_numpy().sum(axis=1, keepdims=True).clip(min=1), axis=0)

    size = max(8.0, n_classes * 0.28)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(normalized.to_numpy(), cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(worst)), worst, rotation=90, fontsize=7)
    ax.set_yticks(range(len(worst)), worst, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{n_classes} least accurate signs, row-normalized")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- Rendering --------------------------------------------------------------

def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render(
    run_name: str,
    split: str,
    joined: pd.DataFrame,
    slots: pd.DataFrame,
    probe: dict[str, object] | None,
    compare: tuple[str, dict[str, object] | None, float] | None,
    per_class: pd.DataFrame,
    pairs: pd.DataFrame,
    figure: Path,
) -> str:
    n = len(joined)
    top1 = float(joined["correct"].mean())
    lo, hi = wilson_interval(int(joined["correct"].sum()), n)
    out: list[str] = ["## Error analysis", ""]
    out.append(
        f"`{run_name}` on the {split} split: {pct(top1)} top-1 "
        f"(95% CI {pct(lo)}-{pct(hi)}), {pct(float(joined['correct5'].mean()))} top-5, "
        f"over {n} sequences from {joined['participant_id'].nunique()} signers."
    )
    out.append("")

    # --- hand slot ---
    out.append("### Which hand the data is in")
    out.append("")
    composition = slots.pivot_table(
        index="split", columns="slot", values="participant_id", aggfunc="count", fill_value=0
    )
    for column in ("left", "mixed", "right"):
        if column not in composition:
            composition[column] = 0
    composition = composition[["left", "mixed", "right"]]
    out.append("| Split | Left slot | Mixed | Right slot |")
    out.append("|---|---|---|---|")
    for split_name in SPLITS:
        if split_name in composition.index:
            row = composition.loc[split_name]
            out.append(f"| {split_name} | {row['left']} | {row['mixed']} | {row['right']} |")
    out.append("")
    out.append(
        "Each signer's one-handed sequences overwhelmingly put the active hand in the "
        "same block, so the block is a stable property of a signer. The training and "
        "evaluation splits do not draw from the same mixture of them."
    )
    out.append("")
    out.append(
        "This describes the recording, not the signer. MediaPipe assigns left and right "
        "from the camera's point of view, so a left-handed signer and a right-handed one "
        "captured in a mirrored frame are indistinguishable here. The model only ever "
        "sees which block is filled, so the distinction does not change anything below."
    )
    out.append("")

    if probe is not None:
        out.append(
            f"Scoring the same sequences mirrored moves top-1 by "
            f"{probe['delta'] * 100:+.2f}pp ({pct(float(probe['top1']))} to "
            f"{pct(float(probe['mirrored_top1']))}). {probe['lost']} sequences are lost and "
            f"{probe['gained']} gained, exact binomial p = {probe['p']:.3g}. "
            f"{pct(float(probe['churn']))} of top-1 predictions change."
        )
        out.append("")
        if compare is not None:
            other_name, other_probe, other_top1 = compare
            if other_probe is not None:
                total = top1 - other_top1
                explained = float(other_probe["delta"])
                out.append(
                    f"Against `{other_name}`, which is identical but trained without "
                    f"augmentation: mirroring moves that model by "
                    f"{explained * 100:+.2f}pp ({pct(other_top1)} to "
                    f"{pct(float(other_probe['mirrored_top1']))}), and "
                    f"{pct(float(other_probe['churn']))} of its predictions change."
                )
                out.append("")
                if total > 0:
                    out.append(
                        f"Augmentation is worth {total * 100:+.2f}pp here. Aligning the "
                        f"unaugmented model's input with the block it was trained on "
                        f"recovers {explained * 100:.2f}pp of that, so roughly "
                        f"{explained / total * 100:.0f}% of the gain is the model no longer "
                        f"depending on which block the data is in, and the remaining "
                        f"{(total - explained) * 100:.2f}pp is augmentation acting as "
                        f"augmentation. Both halves are real; only the first would shrink "
                        f"on an evaluation split that matched the training mixture."
                    )
                    out.append("")

    # --- tracking quality ---
    out.append("### Tracking quality")
    out.append("")
    tracking = joined.copy()
    tracking["bucket"] = pd.cut(
        tracking["active_nan_frac"], bins=TRACKING_BINS, labels=TRACKING_LABELS
    )
    table = group_accuracy(tracking, "bucket", TRACKING_LABELS)
    out.append("| Frames missing on the tracked hand | Sequences | Top-1 | 95% CI |")
    out.append("|---|---|---|---|")
    for _, row in table.iterrows():
        out.append(
            f"| {row['bucket']} | {row['n']} | {pct(row['top1'])} | "
            f"{pct(row['ci_lo'])}-{pct(row['ci_hi'])} |"
        )
    out.append("")
    out.append(
        "The landmark files are not uniformly complete, and the model's input does not "
        "record how much of a sequence was interpolated rather than observed."
    )
    out.append("")

    # --- length ---
    out.append("### Sequence length")
    out.append("")
    length = joined.copy()
    length["bucket"] = pd.cut(length["n_frames"], bins=LENGTH_BINS, labels=LENGTH_LABELS)
    table = group_accuracy(length, "bucket", LENGTH_LABELS)
    out.append("| Frames before resampling | Sequences | Top-1 | 95% CI |")
    out.append("|---|---|---|---|")
    for _, row in table.iterrows():
        out.append(
            f"| {row['bucket']} | {row['n']} | {pct(row['top1'])} | "
            f"{pct(row['ci_lo'])}-{pct(row['ci_hi'])} |"
        )
    out.append("")
    out.append(
        "Sequences longer than the resampling target have frames discarded; shorter ones "
        "are interpolated up. Signing tempo varies by signer, so a length effect and a "
        "signer effect are easy to confuse -- the per-signer breakdown is in the "
        "accompanying CSV."
    )
    out.append("")

    # --- one- vs two-handed ---
    out.append("### One- and two-handed signs")
    out.append("")
    two_handed = int((~joined["one_handed"]).sum())
    per_sign_two = joined.groupby("true", observed=True)["one_handed"].apply(lambda s: 1 - s.mean())
    out.append(
        f"This split has {two_handed} sequences ({pct(two_handed / n)}) in which both hands "
        f"were detected, and no sign reaches {per_sign_two.max() * 100:.0f}% two-handed "
        f"sequences (median {per_sign_two.median() * 100:.1f}%). Signs that are two-handed "
        f"when performed are not two-handed in this data: the non-dominant hand is rarely "
        f"tracked. Whether the model handles two-handed signs worse cannot be answered "
        f"from these landmarks, because there is almost no contrast to measure."
    )
    out.append("")

    # --- collisions ---
    out.append("### Which signs collide")
    out.append("")
    out.append("| Actual | Confused with | -> | <- | Combined rate |")
    out.append("|---|---|---|---|---|")
    for _, row in pairs.iterrows():
        out.append(
            f"| {row['sign_a']} | {row['sign_b']} | {row['a_to_b']} | {row['b_to_a']} "
            f"| {row['symmetric_rate'] * 100:.1f}% |"
        )
    out.append("")
    bidirectional = int(pairs["bidirectional"].sum())
    out.append(
        f"Ranked by the two conditional error rates summed, so a pair appears for being "
        f"mutually confusable rather than for being common. {bidirectional} of "
        f"{len(pairs)} are confused in both directions."
    )
    out.append("")
    out.append(
        "The model sees hand landmarks only. Face and lip landmarks are excluded by "
        "design, so any pair of signs distinguished mainly by mouth shape or other "
        "non-manual markers is not separable by this model in principle rather than "
        "merely in practice."
    )
    out.append("")

    # --- per class ---
    out.append("### Per-class accuracy")
    out.append("")
    quantiles = per_class["top1"].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    median_n = int(per_class["n"].median())
    example_lo, example_hi = wilson_interval(int(round(0.6 * median_n)), median_n)
    out.append(
        f"Across {len(per_class)} signs: min {pct(quantiles[0.0])}, lower quartile "
        f"{pct(quantiles[0.25])}, median {pct(quantiles[0.5])}, upper quartile "
        f"{pct(quantiles[0.75])}, max {pct(quantiles[1.0])}. "
        f"{int((per_class['top1'] == 0).sum())} signs are never predicted correctly."
    )
    out.append("")
    out.append(
        f"These are not ranked, deliberately. The median sign has {median_n} sequences "
        f"in this split, so a sign scoring 60% carries a 95% interval of roughly "
        f"{pct(example_lo)}-{pct(example_hi)}. A table of the worst individual signs "
        f"would mostly be reporting which classes got an unlucky draw, and would not "
        f"reproduce on another test set. Per-sign numbers are in the accompanying CSV "
        f"for anyone who wants them with that caveat attached."
    )
    out.append("")
    # Relative to the report file's own directory: results.md lives in
    # reports/, so a path carrying the reports/ prefix resolves to
    # reports/reports/ and the image silently fails to render.
    out.append(f"![Confusion matrix for the least accurate signs]({figure.name})")
    out.append("")
    return "\n".join(out)


# --- Main -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse a model's errors by sequence property.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument(
        "--compare-run",
        default=None,
        help="a second run to contrast the mirrored scoring against",
    )
    p.add_argument("--top-collisions", type=int, default=TOP_COLLISIONS)
    p.add_argument(
        "--report-path",
        type=Path,
        default=RESULTS_MD,
        help="markdown file to write the section into",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    joined = load_joined(args.run_name, args.split)
    slots = signer_hand_slots(load_all_meta())

    probe = mirror_probe(args.run_name, args.split, joined)
    compare = None
    if args.compare_run:
        other = load_joined(args.compare_run, args.split)
        compare = (
            args.compare_run,
            mirror_probe(args.compare_run, args.split, other),
            float(other["correct"].mean()),
        )

    per_class = (
        joined.groupby("true", observed=True)
        .agg(n=("correct", "size"), top1=("correct", "mean"), top5=("correct5", "mean"))
        .reset_index()
        .rename(columns={"true": "sign"})
    )
    per_class[["ci_lo", "ci_hi"]] = pd.DataFrame(
        [wilson_interval(int(r.top1 * r.n), int(r.n)) for r in per_class.itertuples()],
        index=per_class.index,
    )
    pairs = collisions(joined, args.top_collisions)

    REPORTS.mkdir(exist_ok=True)
    stem = f"{args.run_name}_{args.split}"
    per_class_path = REPORTS / f"per_class_{stem}.csv"
    confusion_path = REPORTS / f"confusion_{stem}.csv"
    figure_path = REPORTS / f"confusion_{stem}.png"
    signers_path = REPORTS / f"signers_{stem}.csv"
    length_path = REPORTS / f"length_by_signer_{stem}.csv"

    per_class.to_csv(per_class_path, index=False, encoding="utf-8")
    pd.crosstab(joined["true"], joined["pred_1"]).to_csv(confusion_path, encoding="utf-8")
    slots.to_csv(signers_path, index=False, encoding="utf-8")

    by_signer = joined.copy()
    by_signer["bucket"] = pd.cut(by_signer["n_frames"], bins=LENGTH_BINS, labels=LENGTH_LABELS)
    (
        by_signer.groupby(["participant_id", "bucket"], observed=True)
        .agg(n=("correct", "size"), top1=("correct", "mean"))
        .reset_index()
        .to_csv(length_path, index=False, encoding="utf-8")
    )
    confusion_figure(joined, figure_path, min(CONFUSION_FIGURE_CLASSES, joined["true"].nunique()))

    section = render(
        args.run_name, args.split, joined, slots, probe, compare, per_class, pairs, figure_path
    )
    # Section name carries the split: analyses of different splits are
    # different results, and a shared name means whichever ran last silently
    # replaces the other.
    report.update_section(args.report_path, f"error_analysis_{args.split}", section)

    for path in (per_class_path, confusion_path, figure_path, signers_path, length_path):
        print(f"wrote {path}")
    print(f"{args.report_path} error_analysis_{args.split} section updated")


if __name__ == "__main__":
    main()
