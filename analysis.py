import csv
import re
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CHANGE THIS PATH
# =========================
csv_path = "/home/iitgn-robotics/Debojit_WS/project_sam/Data/Activity_1.csv"


# =========================
# Robust Vicon CSV loader
# =========================
def load_vicon_csv(path: str) -> pd.DataFrame:
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    frame_idx = None
    for i, line in enumerate(lines):
        s = line.replace("\ufeff", "").strip()
        if re.match(r"^Frame\s*,", s):
            frame_idx = i
            break

    if frame_idx is None or frame_idx < 1:
        raise RuntimeError(
            "Could not find the Vicon header row that starts with 'Frame,'. "
            "Open the CSV and confirm it contains a line like: Frame,Sub Frame,..."
        )

    marker_idx = frame_idx - 1
    data_start_idx = frame_idx + 2  # after units row

    marker_row = next(csv.reader([lines[marker_idx]]))
    axis_row = next(csv.reader([lines[frame_idx]]))

    ncols = max(len(marker_row), len(axis_row))
    marker_row += [None] * (ncols - len(marker_row))
    axis_row += [None] * (ncols - len(axis_row))

    marker_ffill = []
    last = None
    for x in marker_row:
        if x is not None and str(x).strip() != "":
            last = x
        marker_ffill.append(last)

    colnames = []
    for j in range(ncols):
        if j == 0:
            colnames.append("Frame")
        elif j == 1:
            colnames.append("SubFrame")
        else:
            m = marker_ffill[j]
            a = axis_row[j]
            if m is None or a is None or str(a).strip() == "":
                colnames.append(f"col{j}")
            else:
                colnames.append(f"{m}_{a}")

    numeric_text = "".join(lines[data_start_idx:])
    df = pd.read_csv(StringIO(numeric_text), header=None, sep=",", engine="python")

    if df.shape[1] < ncols:
        for k in range(df.shape[1], ncols):
            df[k] = np.nan
    df = df.iloc[:, :ncols]
    df.columns = colnames

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)
    df = df.interpolate(limit=10, limit_direction="both")
    return df


def pts(data: pd.DataFrame, marker: str) -> np.ndarray:
    cols = [f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        available_prefixes = sorted({c.rsplit("_", 1)[0] for c in data.columns if "_" in c})
        raise KeyError(
            f"Missing columns for marker '{marker}': {missing}\n"
            f"Some available marker prefixes: {available_prefixes[:25]}"
        )
    return data[cols].to_numpy(dtype=float)


# =========================
# Load and extract markers
# =========================
data = load_vicon_csv(csv_path)

sho = pts(data, "m10:RSHO")
elb = pts(data, "m10:RELB")
wra = pts(data, "m10:RWRA")
wrb = pts(data, "m10:RWRB")
fin = pts(data, "m10:RFIN")
wrist = 0.5 * (wra + wrb)

frames = data["Frame"].to_numpy(dtype=float)
N = len(frames)
if N == 0:
    raise RuntimeError("No frames found after parsing.")


# =========================
# Choose 6 frames to plot
# =========================
USE_FRAME_NUMBERS = True  # True: list contains Vicon Frame numbers; False: list contains 0-based indices
frame_list = [566, 615, 682, 743, 804, 932]   # if USE_FRAME_NUMBERS=True, these are Vicon Frame numbers


# =========================
# Sagittal plane choice
# True => YZ, False => XZ
# =========================
SAGITTAL_IS_YZ = True


# =========================
# Resolve frame indices
# =========================
def resolve_indices(frame_list_in):
    if not USE_FRAME_NUMBERS:
        idxs = [int(i) for i in frame_list_in]
    else:
        frame_to_idx = {}
        for idx, fr in enumerate(frames):
            if np.isfinite(fr):
                frame_to_idx[int(fr)] = idx

        idxs = []
        missing = []
        for fr in frame_list_in:
            fr_int = int(fr)
            if fr_int in frame_to_idx:
                idxs.append(frame_to_idx[fr_int])
            else:
                missing.append(fr_int)
        if missing:
            raise ValueError(f"These frame numbers were not found in the CSV: {missing}")

    for i in idxs:
        if i < 0 or i >= N:
            raise ValueError(f"Frame index out of range: {i} (valid 0..{N-1})")
    return idxs


idxs = resolve_indices(frame_list)


# =========================
# Global limits (consistent across subplots)
# =========================
all_pts = np.vstack([sho, elb, wrist, fin])
mins = np.nanmin(all_pts, axis=0)
maxs = np.nanmax(all_pts, axis=0)
center = 0.5 * (mins + maxs)
span = np.nanmax(maxs - mins)
pad = 0.1 * span if np.isfinite(span) and span > 0 else 1.0

if SAGITTAL_IS_YZ:
    xlim = (center[1] - span / 2 - pad, center[1] + span / 2 + pad)  # Y range
    ylim = (center[2] - span / 2 - pad, center[2] + span / 2 + pad)  # Z range
    xlabel, ylabel = "Y (mm)", "Z (mm)"
else:
    xlim = (center[0] - span / 2 - pad, center[0] + span / 2 + pad)  # X range
    ylim = (center[2] - span / 2 - pad, center[2] + span / 2 + pad)  # Z range
    xlabel, ylabel = "X (mm)", "Z (mm)"


# =========================
# Colors and labels
# =========================
COL_SHO = "tab:blue"
COL_ELB = "tab:orange"
COL_WRI = "tab:green"
COL_FIN = "tab:red"
COL_LINK = "k"

POINT_LABELS = ["SHO", "ELB", "WRI", "FIN"]
POINT_COLS = [COL_SHO, COL_ELB, COL_WRI, COL_FIN]


# =========================
# Helpers for angle + coordinate readout
# =========================
def elbow_to_wrist_angle_deg(i: int) -> float:
    """
    Angle of vector (ELB -> WRI) w.r.t. horizontal axis in the sagittal plane.
    If SAGITTAL_IS_YZ: horizontal = +Y, vertical = +Z, angle = atan2(dZ, dY)
    If SAGITTAL_IS_YZ is False (XZ): horizontal = +X, vertical = +Z, angle = atan2(dZ, dX)
    """
    if SAGITTAL_IS_YZ:
        dy = wrist[i, 1] - elb[i, 1]
        dz = wrist[i, 2] - elb[i, 2]
        ang = np.degrees(np.arctan2(dz, dy))
    else:
        dx = wrist[i, 0] - elb[i, 0]
        dz = wrist[i, 2] - elb[i, 2]
        ang = np.degrees(np.arctan2(dz, dx))
    return float(ang)


def wrist_coords_in_plane(i: int):
    if SAGITTAL_IS_YZ:
        return float(wrist[i, 1]), float(wrist[i, 2])  # (Y, Z)
    return float(wrist[i, 0]), float(wrist[i, 2])      # (X, Z)


# =========================
# Plot 2 x 3 sagittal snapshots
# =========================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

# Make one legend for the whole figure (not repeated per subplot)
legend_handles = []
legend_labels = []
for lab, col in zip(POINT_LABELS, POINT_COLS):
    h = plt.Line2D([], [], marker="o", linestyle="None", color=col, markersize=7)
    legend_handles.append(h)
    legend_labels.append(lab)
legend_handles.append(plt.Line2D([], [], color=COL_LINK, linewidth=3))
legend_labels.append("Links")

for ax, i in zip(axes, idxs):
    # Extract sagittal coordinates
    if SAGITTAL_IS_YZ:
        xs = [sho[i, 1], elb[i, 1], wrist[i, 1], fin[i, 1]]
        ys = [sho[i, 2], elb[i, 2], wrist[i, 2], fin[i, 2]]
    else:
        xs = [sho[i, 0], elb[i, 0], wrist[i, 0], fin[i, 0]]
        ys = [sho[i, 2], elb[i, 2], wrist[i, 2], fin[i, 2]]

    # Links
    ax.plot(xs, ys, linewidth=3, color=COL_LINK)

    # Dots
    ax.scatter(xs[0], ys[0], s=50, color=COL_SHO)
    ax.scatter(xs[1], ys[1], s=50, color=COL_ELB)
    ax.scatter(xs[2], ys[2], s=50, color=COL_WRI)
    ax.scatter(xs[3], ys[3], s=50, color=COL_FIN)

    # Point labels next to dots
    # Small offset relative to axis span for readability
    dx_txt = 0.01 * (xlim[1] - xlim[0])
    dy_txt = 0.01 * (ylim[1] - ylim[0])
    for (x, y, lab) in zip(xs, ys, POINT_LABELS):
        ax.text(x + dx_txt, y + dy_txt, lab, fontsize=10)

    # Angle and wrist coordinate readout
    ang = elbow_to_wrist_angle_deg(i)
    wx, wz = wrist_coords_in_plane(i)
    if SAGITTAL_IS_YZ:
        coord_str = f"WRI (Y,Z) = ({wx:.1f}, {wz:.1f})"
    else:
        coord_str = f"WRI (X,Z) = ({wx:.1f}, {wz:.1f})"
    info_str = f"Angle(ELB→WRI vs horizontal) = {ang:.1f}°\n{coord_str}"
    ax.text(
        0.02, 0.98, info_str,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Index {i} | Frame {int(frames[i])}")

# Hide unused axes if fewer than 6 indices
for ax in axes[len(idxs):]:
    ax.axis("off")

# Global legend
fig.legend(legend_handles, legend_labels, loc="lower center", ncol=5, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend at bottom
plt.show()