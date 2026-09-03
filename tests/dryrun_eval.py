"""Dry run for the metadata pass and the evaluation harness.

Builds a small synthetic dataset shaped like the real one (parquet files in
the same long format, a train.csv, a splits.json), runs the real scripts
over it as subprocesses, and checks the outputs. Nothing here touches the
real dataset.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import preprocessing as pp  # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, condition: bool) -> None:
    CHECKS.append((name, bool(condition)))


def make_sequence(n_frames: int, left_missing: float, right_missing: float, seed: int) -> pd.DataFrame:
    """One sequence in the raw long format: frame x landmark rows, with
    per-hand NaN rates controlled so the unused-hand rule can be exercised."""
    rng = np.random.default_rng(seed)
    rows = []
    for frame in range(n_frames):
        for kind, count, missing in (
            ("left_hand", 21, left_missing),
            ("right_hand", 21, right_missing),
            ("pose", 33, 0.0),
            ("face", 5, 0.0),
        ):
            drop = rng.random() < missing
            for idx in range(count):
                if kind == "pose" and idx in (11, 12):
                    x, y, z = (0.4, 0.5, 0.0) if idx == 11 else (0.6, 0.5, 0.0)
                elif drop:
                    x = y = z = np.nan
                else:
                    x, y, z = rng.random(3)
                rows.append(
                    {
                        "frame": frame,
                        "row_id": f"{frame}-{kind}-{idx}",
                        "type": kind,
                        "landmark_index": idx,
                        "x": x,
                        "y": y,
                        "z": z,
                    }
                )
    return pd.DataFrame(rows)


def build_dataset(root: Path) -> None:
    raw = root / "data" / "raw" / "train_landmark_files"
    raw.mkdir(parents=True, exist_ok=True)

    signs = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    # participant -> (n sequences). Two signers per split so every split has
    # more than one, and enough sequences that filters have something to bite.
    plan = {10: "train", 11: "train", 20: "val", 30: "test"}
    records = []
    seed = 0
    for participant, _ in plan.items():
        (raw / str(participant)).mkdir(exist_ok=True)
        for i in range(18):
            seed += 1
            sign = signs[i % len(signs)]
            # A few deliberate filter cases: one too-short, one with both
            # hands absent, one long sequence, and one one-handed sequence.
            if i == 0:
                n_frames, lm, rm = 2, 0.0, 0.0  # too short
            elif i == 1:
                n_frames, lm, rm = 10, 1.0, 1.0  # both hands unused
            elif i == 2:
                n_frames, lm, rm = 90, 1.0, 0.0  # long, right hand only
            elif i == 3:
                n_frames, lm, rm = 15, 0.0, 1.0  # left hand only
            else:
                n_frames, lm, rm = 10 + i, 0.2, 0.1
            df = make_sequence(n_frames, lm, rm, seed)
            rel = f"train_landmark_files/{participant}/{seed}.parquet"
            df.to_parquet(root / "data" / "raw" / rel)
            records.append(
                {
                    "path": rel,
                    "participant_id": participant,
                    "sequence_id": seed,
                    "sign": sign,
                }
            )

    pd.DataFrame(records).to_csv(root / "data" / "raw" / "train.csv", index=False)
    (root / "data" / "raw" / "sign_to_prediction_index_map.json").write_text(
        json.dumps({s: i for i, s in enumerate(signs)}), encoding="utf-8"
    )
    splits = {"train": [10, 11], "val": [20], "test": [30]}
    (root / "data" / "splits.json").write_text(json.dumps(splits), encoding="utf-8")


def run(root: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke one of the scripts the way a person would, in a throwaway
    working directory.

    The environment is inherited and only added to. Replacing it outright
    strands the interpreter without the variables the platform's own
    libraries need -- on Windows, dropping SystemRoot stops Winsock
    initializing, which surfaces as an unrelated-looking import error deep
    inside torch."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, str(SRC / args[0]), *args[1:]],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )


def main() -> int:
    # ignore_cleanup_errors: worker processes on Windows can still hold
    # handles on the scratch files when the block exits, and a cleanup
    # failure there would mask the check results.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "reports").mkdir()
        (root / "models").mkdir()
        build_dataset(root)

        # --- full cache build, which should also emit metadata -------------
        r = run(root, "cache_dataset.py")
        check("cache build succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            return 1

        cache_dir = root / "data" / "cache"
        for split in ("train", "val", "test"):
            check(f"{split} metadata written by full build", (cache_dir / f"meta_{split}.csv").exists())

        meta = pd.read_csv(cache_dir / "meta_test.csv")
        y = np.load(cache_dir / "normalized_test_y.npy", allow_pickle=False)
        x = np.load(cache_dir / "normalized_test_X.npy")

        check("metadata rows == cached label rows", len(meta) == len(y))
        check("metadata rows == cached array rows", len(meta) == len(x))
        check("metadata sign column matches cached labels", list(meta["sign"]) == list(y))
        check("filters dropped exactly two sequences per signer", len(meta) == 16)
        check("metadata columns as specified", list(meta.columns) == [
            "participant_id", "sequence_id", "sign", "n_frames",
            "left_hand_nan_frac", "right_hand_nan_frac",
            "left_hand_unused", "right_hand_unused",
        ])
        check("participant ids are the split's signers", set(meta["participant_id"]) == {30})
        check("sequence ids unique", meta["sequence_id"].is_unique)

        # Filter cases: too-short and both-hands-unused must be gone, and the
        # one-handed cases must be present and correctly flagged.
        check("too-short sequence excluded", 2 not in set(meta["n_frames"]))
        check("both-hands-unused excluded", not ((meta["left_hand_unused"] == 1) & (meta["right_hand_unused"] == 1)).any())
        check("long sequence retained with true frame count", 90 in set(meta["n_frames"]))
        right_only = meta[(meta["left_hand_unused"] == 1) & (meta["right_hand_unused"] == 0)]
        left_only = meta[(meta["right_hand_unused"] == 1) & (meta["left_hand_unused"] == 0)]
        check("right-hand-only sequence flagged", len(right_only) == 1)
        check("left-hand-only sequence flagged", len(left_only) == 1)
        check("frame count is pre-resample, not TARGET_LEN", set(meta["n_frames"]) != {pp.TARGET_LEN})
        check("nan fracs within [0,1]", meta["left_hand_nan_frac"].between(0, 1).all())

        # An unused hand must be exactly zero in the cached array, which is
        # what makes "hand absent" distinguishable from a coordinate.
        row = int(right_only.index[0])
        left_block = x[row, :, : len(pp.HAND_INDICES), :]
        check("unused hand is exactly zero in cached array", np.all(left_block == 0.0))

        # --- metadata-only mode -------------------------------------------
        before = {p.name: p.stat().st_mtime_ns for p in cache_dir.glob("*.npy")}
        r = run(root, "cache_dataset.py", "--metadata-only")
        check("metadata-only succeeds", r.returncode == 0)
        check("metadata-only verified alignment", "match" in r.stdout)
        after = {p.name: p.stat().st_mtime_ns for p in cache_dir.glob("*.npy")}
        check("metadata-only leaves arrays untouched", before == after)
        meta2 = pd.read_csv(cache_dir / "meta_test.csv")
        check("metadata-only reproduces the full build's metadata", meta.equals(meta2))

        # --- alignment check actually catches a misalignment ---------------
        good = (cache_dir / "normalized_test_y.npy").read_bytes()
        scrambled = np.load(cache_dir / "normalized_test_y.npy", allow_pickle=False)[::-1].copy()
        np.save(cache_dir / "normalized_test_y.npy", scrambled)
        r = run(root, "cache_dataset.py", "--metadata-only")
        check("misaligned labels are detected", r.returncode != 0 and "disagree" in (r.stdout + r.stderr))
        truncated = np.load(cache_dir / "normalized_test_y.npy", allow_pickle=False)[:-1]
        np.save(cache_dir / "normalized_test_y.npy", truncated)
        r = run(root, "cache_dataset.py", "--metadata-only")
        check("row-count mismatch is detected", r.returncode != 0 and "rows" in (r.stdout + r.stderr))
        (cache_dir / "normalized_test_y.npy").write_bytes(good)

        # --- train a tiny model so there is a checkpoint to evaluate -------
        r = run(root, "train.py", "--model", "cnn", "--landmarks", "hands",
                "--epochs", "2", "--batch-size", "8", "--run-name", "dry_cnn")
        check("training a checkpoint succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            return 1

        # --- evaluate ------------------------------------------------------
        r = run(root, "evaluate.py", "--run-name", "dry_cnn", "--split", "test")
        check("evaluate succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            return 1
        check("evaluate reports reproducing the recorded score", "Reproduces the recorded" in r.stdout)

        preds = pd.read_csv(root / "reports" / "preds_dry_cnn_test.csv")
        check("one prediction row per test sequence", len(preds) == len(y))
        check("prediction rows align with metadata order", list(preds["true"]) == list(meta["sign"]))
        check("prediction columns as specified", list(preds.columns) == [
            "row", "true", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5",
            "true_rank", "top1_prob", "true_prob",
        ])
        check("true_rank >= 1", (preds["true_rank"] >= 1).all())
        check("rank 1 iff pred_1 is the true sign",
              ((preds["true_rank"] == 1) == (preds["pred_1"] == preds["true"])).all())
        check("probabilities in [0,1]", preds["top1_prob"].between(0, 1).all() and preds["true_prob"].between(0, 1).all())
        check("top1_prob >= true_prob", (preds["top1_prob"] >= preds["true_prob"] - 1e-6).all())
        check("predicted signs are in the vocabulary", set(preds["pred_1"]) <= set(y))

        runs = pd.read_csv(root / "reports" / "ablations.csv")
        recorded = float(runs[runs["run_name"] == "dry_cnn"]["test_top1"].iloc[0])
        measured = float((preds["true_rank"] == 1).mean())
        check("evaluate reproduces the recorded test top-1", abs(recorded - measured) < 5e-5)

        # --- mirrored scoring ---------------------------------------------
        r = run(root, "evaluate.py", "--run-name", "dry_cnn", "--split", "test", "--mirror")
        check("mirrored evaluate succeeds", r.returncode == 0)
        mpath = root / "reports" / "preds_dry_cnn_test_mirrored.csv"
        check("mirrored predictions go to their own file", mpath.exists())
        mirrored = pd.read_csv(mpath)
        check("mirrored file has the same rows", len(mirrored) == len(preds))
        check("mirrored true labels unchanged", list(mirrored["true"]) == list(preds["true"]))
        check("mirroring changes at least one prediction",
              not mirrored["pred_1"].equals(preds["pred_1"]) or not np.allclose(mirrored["true_prob"], preds["true_prob"]))
        check("unmirrored file not overwritten by mirrored run",
              len(pd.read_csv(root / "reports" / "preds_dry_cnn_test.csv")) == len(preds))

        # --- config is taken from the checkpoint, not from flags -----------
        # dry_cnn was trained on hands only; evaluate must slice the cache the
        # same way rather than feeding it all 50 landmarks.
        check("evaluate used the checkpoint's landmark set", "landmarks: hands" in r.stdout)

        # --- val split -----------------------------------------------------
        r = run(root, "evaluate.py", "--run-name", "dry_cnn", "--split", "val")
        check("val split evaluates", r.returncode == 0)
        check("val predictions written", (root / "reports" / "preds_dry_cnn_val.csv").exists())

        # --- missing checkpoint fails clearly ------------------------------
        r = run(root, "evaluate.py", "--run-name", "not_a_run", "--split", "test")
        check("missing checkpoint reports the run it needs", r.returncode != 0 and "not_a_run" in (r.stdout + r.stderr))


        # --- error analysis -------------------------------------------------
        r = run(root, "error_analysis.py", "--run-name", "dry_cnn", "--split", "test",
                "--compare-run", "dry_cnn", "--top-collisions", "5")
        check("error analysis succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-4000:], r.stderr[-4000:])
            return 1

        rep = root / "reports"
        for name in ("per_class_dry_cnn_test.csv", "confusion_dry_cnn_test.csv",
                     "confusion_dry_cnn_test.png", "signers_dry_cnn_test.csv",
                     "length_by_signer_dry_cnn_test.csv"):
            check(f"error analysis wrote {name}", (rep / name).exists())

        pc = pd.read_csv(rep / "per_class_dry_cnn_test.csv")
        check("per-class covers every sign present", len(pc) == preds["true"].nunique())
        check("per-class counts sum to the split", pc["n"].sum() == len(preds))
        check("per-class accuracy in [0,1]", pc["top1"].between(0, 1).all())
        check("per-class CI brackets the estimate",
              ((pc["ci_lo"] <= pc["top1"] + 1e-9) & (pc["top1"] <= pc["ci_hi"] + 1e-9)).all())
        check("per-class top5 >= top1", (pc["top5"] >= pc["top1"] - 1e-9).all())

        conf = pd.read_csv(rep / "confusion_dry_cnn_test.csv", index_col=0)
        check("confusion matrix totals the split", conf.to_numpy().sum() == len(preds))
        diag = sum(conf.loc[s, s] for s in conf.index if s in conf.columns)
        check("confusion diagonal equals correct count", diag == int((preds["true_rank"] == 1).sum()))

        sg = pd.read_csv(rep / "signers_dry_cnn_test.csv")
        check("signer table covers all signers", len(sg) == 4)
        check("signer slots labelled", set(sg["slot"]) <= {"left", "mixed", "right"})
        check("pct_right within [0,100]", sg["pct_right"].between(0, 100).all())

        results = (rep / "results.md").read_text(encoding="utf-8")
        check("error analysis section written", "<!-- ERROR_ANALYSIS_TEST_START -->" in results)
        check("runs section still present", "<!-- RUNS_START -->" in results)
        check("mirror probe reported", "mirrored" in results.lower())
        check("per-class ranking caveat present", "not ranked" in results)

        # Bin labels are ranges that do not sort as text; a lexicographic
        # ordering would present a monotone trend as noise.
        def bins_ordered(section_start, labels):
            """Bins present in the table must appear in range order. A small
            split need not populate every bin, so absent ones are skipped."""
            body = results.split(section_start)[1].split("###")[0]
            positions = {l: body.index("| " + l + " |") for l in labels if "| " + l + " |" in body}
            present = [l for l in labels if l in positions]
            by_position = [l for l, _ in sorted(positions.items(), key=lambda kv: kv[1])]
            return len(present) >= 2 and present == by_position
        check("length bins in range order",
              bins_ordered("### Sequence length", ["<=22", "23-70", "71-135", ">135"]))
        check("tracking bins in range order",
              bins_ordered("### Tracking quality", ["<10%", "10-25%", "25-50%", ">50%"]))

        # Rerunning must replace its own section, not append a second one.
        r = run(root, "error_analysis.py", "--run-name", "dry_cnn", "--split", "test",
                "--top-collisions", "5")
        check("error analysis rerun succeeds", r.returncode == 0)
        again = (rep / "results.md").read_text(encoding="utf-8")
        check("section not duplicated on rerun", again.count("<!-- ERROR_ANALYSIS_TEST_START -->") == 1)
        check("runs section survives rerun", "<!-- RUNS_START -->" in again)

        # Analysing a second split must not replace the first split's section.
        r = run(root, "error_analysis.py", "--run-name", "dry_cnn", "--split", "val",
                "--top-collisions", "3")
        check("val split analysis succeeds", r.returncode == 0)
        both = (rep / "results.md").read_text(encoding="utf-8")
        check("test section survives a val run", "<!-- ERROR_ANALYSIS_TEST_START -->" in both)
        check("val section added alongside", "<!-- ERROR_ANALYSIS_VAL_START -->" in both)

        # results.md lives in reports/, so the figure link must not carry a
        # reports/ prefix or it resolves to reports/reports/.
        check("figure link is relative to the report file",
              "](confusion_dry_cnn_test.png)" in both and "](reports/" not in both)

        # A separate destination keeps diagnostic splits out of the headline file.
        r = run(root, "error_analysis.py", "--run-name", "dry_cnn", "--split", "val",
                "--top-collisions", "3", "--report-path", "reports/diagnostics.md")
        check("alternate report path succeeds", r.returncode == 0)
        check("alternate report file written", (rep / "diagnostics.md").exists())

        # --- latency --------------------------------------------------------
        r = run(root, "latency.py", "--run-name", "dry_cnn",
                "--warmup", "3", "--iterations", "20")
        check("latency benchmark succeeds", r.returncode == 0)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            return 1
        lat = (rep / "results.md").read_text(encoding="utf-8")
        check("latency section written", "<!-- LATENCY_START -->" in lat)
        check("latency reports both thread counts", "1 |" in lat and "(default)" in lat)
        check("latency did not clobber the error analysis",
              "<!-- ERROR_ANALYSIS_TEST_START -->" in lat)
        import re as _re
        ms = [float(v) for v in _re.findall(r"([0-9.]+) ms", lat)]
        check("latency values are positive", len(ms) >= 10 and all(v > 0 for v in ms))
        # Percentiles are order statistics of the same sample, so this
        # ordering must hold for any input; a violation means the summary
        # is indexing the sorted timings wrongly.
        rows = _re.findall(r"\| ([0-9.]+) ms \| ([0-9.]+) ms \| ([0-9.]+) ms \| ([0-9.]+) ms \| ([0-9.]+) ms \|", lat)
        check("percentiles ordered within each row",
              len(rows) == 2 and all(
                  float(mn) <= float(med) <= float(p95) <= float(p99) <= float(mx)
                  for med, p95, p99, mn, mx in rows))

        # A predictions file that does not line up with the metadata must fail.
        bad = pd.read_csv(rep / "preds_dry_cnn_test.csv")
        bad["true"] = bad["true"].iloc[::-1].to_numpy()
        bad.to_csv(rep / "preds_dry_cnn_test.csv", index=False)
        r = run(root, "error_analysis.py", "--run-name", "dry_cnn", "--split", "test")
        check("misaligned predictions rejected",
              r.returncode != 0 and "row orders do not match" in (r.stdout + r.stderr))

        # --- a drifted checkpoint is caught --------------------------------
        runs.loc[runs["run_name"] == "dry_cnn", "test_top1"] = 0.9999
        runs.to_csv(root / "reports" / "ablations.csv", index=False)
        r = run(root, "evaluate.py", "--run-name", "dry_cnn", "--split", "test")
        check("recorded-score mismatch fails loudly",
              r.returncode != 0 and "not comparable" in (r.stdout + r.stderr))

    passed = sum(1 for _, ok in CHECKS if ok)
    for name, ok in CHECKS:
        if not ok:
            print(f"FAIL  {name}")
    print(f"\n{passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
