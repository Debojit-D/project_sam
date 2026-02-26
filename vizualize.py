import csv
import re
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================
# CHANGE THIS PATH
# =========================
csv_path = "/home/debojit/Debojit_WS/project_sam/Data/Activity_1.csv"


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


# =========================
# Load data
# =========================
data = load_vicon_csv(csv_path)


# =========================
# Helper to extract markers
# =========================
def pts(marker: str) -> np.ndarray:
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
# Extract arm markers
# =========================
sho = pts("m10:RSHO")
elb = pts("m10:RELB")
wra = pts("m10:RWRA")
wrb = pts("m10:RWRB")
fin = pts("m10:RFIN")
wrist = 0.5 * (wra + wrb)

frames = data["Frame"].to_numpy(dtype=float)
N = len(frames)
if N == 0:
    raise RuntimeError("No frames found after parsing. Check the CSV content and header alignment.")


# =========================
# Plot setup
# =========================
plt.close("all")
fig = plt.figure(figsize=(14, 7))
ax3d = fig.add_subplot(121, projection="3d")
ax2d = fig.add_subplot(122)
plt.subplots_adjust(bottom=0.18)

# Consistent colors for the same joint across views
COL_SHO = "tab:blue"
COL_ELB = "tab:orange"
COL_WRI = "tab:green"
COL_FIN = "tab:red"
COL_TRAIL = "tab:purple"

# =========================
# 3D axis
# =========================
ax3d.set_title("Right Arm Kinematic Reconstruction (3D)")
ax3d.set_xlabel("X (mm)")
ax3d.set_ylabel("Y (mm)")
ax3d.set_zlabel("Z (mm)")

all_pts = np.vstack([sho, elb, wrist, fin])
mins = np.nanmin(all_pts, axis=0)
maxs = np.nanmax(all_pts, axis=0)
center = 0.5 * (mins + maxs)
span = np.nanmax(maxs - mins)
pad = 0.1 * span if np.isfinite(span) and span > 0 else 1.0

ax3d.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
ax3d.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
ax3d.set_zlim(center[2] - span / 2 - pad, center[2] + span / 2 + pad)

try:
    ax3d.set_box_aspect((1, 1, 1))
except Exception:
    pass

# =========================
# 2D axis (YZ sagittal view)
# =========================
SAGITTAL_IS_YZ = True  # True => YZ, False => XZ

if SAGITTAL_IS_YZ:
    ax2d.set_title("Sagittal View (Y-Z)")
    ax2d.set_xlabel("Y (mm)")
    ax2d.set_ylabel("Z (mm)")
    ax2d.set_aspect("equal")
    ax2d.set_xlim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax2d.set_ylim(center[2] - span / 2 - pad, center[2] + span / 2 + pad)
else:
    ax2d.set_title("Sagittal View (X-Z)")
    ax2d.set_xlabel("X (mm)")
    ax2d.set_ylabel("Z (mm)")
    ax2d.set_aspect("equal")
    ax2d.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax2d.set_ylim(center[2] - span / 2 - pad, center[2] + span / 2 + pad)


# =========================
# Initial frame
# =========================
i0 = 0

# 3D links
link3d, = ax3d.plot(
    [sho[i0, 0], elb[i0, 0], wrist[i0, 0], fin[i0, 0]],
    [sho[i0, 1], elb[i0, 1], wrist[i0, 1], fin[i0, 1]],
    [sho[i0, 2], elb[i0, 2], wrist[i0, 2], fin[i0, 2]],
    linewidth=3,
    color="k",
    label="Arm Links",
)

# 3D dots (force same colors)
shoulder3d = ax3d.scatter(sho[i0, 0], sho[i0, 1], sho[i0, 2], s=60, color=COL_SHO, label="Shoulder")
elbow3d = ax3d.scatter(elb[i0, 0], elb[i0, 1], elb[i0, 2], s=60, color=COL_ELB, label="Elbow")
wrist3d = ax3d.scatter(wrist[i0, 0], wrist[i0, 1], wrist[i0, 2], s=60, color=COL_WRI, label="Wrist")
finger3d = ax3d.scatter(fin[i0, 0], fin[i0, 1], fin[i0, 2], s=60, color=COL_FIN, label="Finger")

# 3D trajectory
trail3d, = ax3d.plot(
    fin[:i0 + 1, 0],
    fin[:i0 + 1, 1],
    fin[:i0 + 1, 2],
    linewidth=1,
    color=COL_TRAIL,
    label="Finger Trajectory",
)

ax3d.legend(loc="upper left")

# 2D links + dots + trajectory (use the same colors)
if SAGITTAL_IS_YZ:
    link2d, = ax2d.plot(
        [sho[i0, 1], elb[i0, 1], wrist[i0, 1], fin[i0, 1]],
        [sho[i0, 2], elb[i0, 2], wrist[i0, 2], fin[i0, 2]],
        linewidth=3,
        color="k",
    )
    shoulder2d, = ax2d.plot(sho[i0, 1], sho[i0, 2], "o", color=COL_SHO)
    elbow2d, = ax2d.plot(elb[i0, 1], elb[i0, 2], "o", color=COL_ELB)
    wrist2d, = ax2d.plot(wrist[i0, 1], wrist[i0, 2], "o", color=COL_WRI)
    finger2d, = ax2d.plot(fin[i0, 1], fin[i0, 2], "o", color=COL_FIN)

    trail2d, = ax2d.plot(fin[:i0 + 1, 1], fin[:i0 + 1, 2], linewidth=1, color=COL_TRAIL)
else:
    link2d, = ax2d.plot(
        [sho[i0, 0], elb[i0, 0], wrist[i0, 0], fin[i0, 0]],
        [sho[i0, 2], elb[i0, 2], wrist[i0, 2], fin[i0, 2]],
        linewidth=3,
        color="k",
    )
    shoulder2d, = ax2d.plot(sho[i0, 0], sho[i0, 2], "o", color=COL_SHO)
    elbow2d, = ax2d.plot(elb[i0, 0], elb[i0, 2], "o", color=COL_ELB)
    wrist2d, = ax2d.plot(wrist[i0, 0], wrist[i0, 2], "o", color=COL_WRI)
    finger2d, = ax2d.plot(fin[i0, 0], fin[i0, 2], "o", color=COL_FIN)

    trail2d, = ax2d.plot(fin[:i0 + 1, 0], fin[:i0 + 1, 2], linewidth=1, color=COL_TRAIL)

info = fig.text(0.02, 0.95, "")

# =========================
# Slider
# =========================
ax_slider = fig.add_axes([0.2, 0.06, 0.6, 0.04])
slider = Slider(ax_slider, "Frame", 0, N - 1, valinit=i0, valstep=1)


# =========================
# Update function
# =========================
def update(val):
    i = int(slider.val)

    # 3D link
    link3d.set_data(
        [sho[i, 0], elb[i, 0], wrist[i, 0], fin[i, 0]],
        [sho[i, 1], elb[i, 1], wrist[i, 1], fin[i, 1]],
    )
    link3d.set_3d_properties([sho[i, 2], elb[i, 2], wrist[i, 2], fin[i, 2]])

    # 3D dots
    shoulder3d._offsets3d = ([sho[i, 0]], [sho[i, 1]], [sho[i, 2]])
    elbow3d._offsets3d = ([elb[i, 0]], [elb[i, 1]], [elb[i, 2]])
    wrist3d._offsets3d = ([wrist[i, 0]], [wrist[i, 1]], [wrist[i, 2]])
    finger3d._offsets3d = ([fin[i, 0]], [fin[i, 1]], [fin[i, 2]])

    # 3D trail
    trail3d.set_data(fin[:i + 1, 0], fin[:i + 1, 1])
    trail3d.set_3d_properties(fin[:i + 1, 2])

    # 2D link + dots + trail
    if SAGITTAL_IS_YZ:
        link2d.set_data(
            [sho[i, 1], elb[i, 1], wrist[i, 1], fin[i, 1]],
            [sho[i, 2], elb[i, 2], wrist[i, 2], fin[i, 2]],
        )
        shoulder2d.set_data([sho[i, 1]], [sho[i, 2]])
        elbow2d.set_data([elb[i, 1]], [elb[i, 2]])
        wrist2d.set_data([wrist[i, 1]], [wrist[i, 2]])
        finger2d.set_data([fin[i, 1]], [fin[i, 2]])
        trail2d.set_data(fin[:i + 1, 1], fin[:i + 1, 2])
    else:
        link2d.set_data(
            [sho[i, 0], elb[i, 0], wrist[i, 0], fin[i, 0]],
            [sho[i, 2], elb[i, 2], wrist[i, 2], fin[i, 2]],
        )
        shoulder2d.set_data([sho[i, 0]], [sho[i, 2]])
        elbow2d.set_data([elb[i, 0]], [elb[i, 2]])
        wrist2d.set_data([wrist[i, 0]], [wrist[i, 2]])
        finger2d.set_data([fin[i, 0]], [fin[i, 2]])
        trail2d.set_data(fin[:i + 1, 0], fin[:i + 1, 2])

    info.set_text(f"Index: {i} | Frame: {int(frames[i])}")
    fig.canvas.draw_idle()


slider.on_changed(update)
info.set_text(f"Index: {i0} | Frame: {int(frames[i0])}")

plt.show()