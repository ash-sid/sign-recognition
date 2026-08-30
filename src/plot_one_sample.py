"""
src/plot_one_sample.py

Loads a single landmark sequence and plots the first frame, colored by
landmark type (face/pose/left_hand/right_hand). Sanity check that the
parquet files can be read and look like sane landmark data.

Run: python src\\plot_one_sample.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW = Path("data/raw")

train = pd.read_csv(RAW / "train.csv")
row = train.iloc[0]
print(f"Plotting sign='{row['sign']}' participant={row['participant_id']} sequence={row['sequence_id']}")

seq = pd.read_parquet(RAW / row["path"])

# plot the first frame, colored by landmark type
first_frame = seq[seq["frame"] == seq["frame"].min()]

fig, ax = plt.subplots(figsize=(6, 6))
for landmark_type, group in first_frame.groupby("type"):
    ax.scatter(group["x"], -group["y"], s=8, label=landmark_type, alpha=0.7)

ax.set_title(f"sign='{row['sign']}' — frame {int(first_frame['frame'].iloc[0])}")
ax.legend()
ax.set_aspect("equal")
out_path = Path("reports") / "sample_landmark_plot.png"
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=120)
print(f"Saved to {out_path}")
