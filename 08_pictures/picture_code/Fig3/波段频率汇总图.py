from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import path as mpath
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

from selection_records import SELECTION_RECORDS


# ==========================================
# 1. Plot configuration
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "original.xlsx"
OUTPUT_FILE = SCRIPT_DIR / "band_selection_frequency_summary.svg"

FONT_FAMILY = "Arial"
FIG_SIZE = (7.5, 6)
FONT_SIZE_LABEL = 30
FONT_SIZE_SECONDARY_LABEL = 30
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 17
FREQ_PANEL_HEIGHT_RATIO = 1.55
MIRROR_BAR_WIDTH = 9.5
MIRROR_BAR_ALPHA = 0.88
TOTAL_FOLDS = 5

SN_COLOR = "#C73E3A"
BI_COLOR = "#2F6FB0"
MEAN_COLOR = "#202020"
BASELINE_COLOR = "#777777"

ZONES = [
    {"name": "LFR", "range": (400, 900), "color": "#D8E7F5"},
    {"name": "MFR", "range": (900, 1400), "color": "#DDEDD8"},
    {"name": "HFR", "range": (1400, 1800), "color": "#FFF1C7"},
]


# ==========================================
# 2. Data processing
# ==========================================
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})


def load_mean_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the original spectra and return wavenumber and mean intensity."""
    try:
        df = pd.read_excel(path, usecols="FN:AGR", header=0)
        wavelengths = df.columns.astype(float).to_numpy()
        mean_spectrum = df.mean(axis=0).to_numpy()
        print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} bands")
        return wavelengths, mean_spectrum
    except Exception as exc:
        print(f"Failed to read {path}. Demo data are used instead. ({exc})")
        wavelengths = np.linspace(400, 1800, 700)
        mean_spectrum = (
            1.00 * np.exp(-((wavelengths - 550) ** 2) / (2 * 60 ** 2))
            + 0.80 * np.exp(-((wavelengths - 850) ** 2) / (2 * 80 ** 2))
            + 0.60 * np.exp(-((wavelengths - 1250) ** 2) / (2 * 70 ** 2))
            + 0.30
        )
        return wavelengths, mean_spectrum


def summarize_frequency(records: list[dict], metal: str) -> tuple[np.ndarray, np.ndarray]:
    """Count how many folds selected each wavelength for one target metal."""
    counter: Counter[float] = Counter()
    for record in records:
        if record["metal"] == metal:
            counter.update(record["wavelengths"])

    waves = np.array(sorted(counter), dtype=float)
    counts = np.array([counter[wave] for wave in waves], dtype=float)
    return waves, counts


def draw_mirrored_frequency_bars(
    axis: plt.Axes,
    waves: np.ndarray,
    counts: np.ndarray,
    direction: float,
    color: str,
    width: float = MIRROR_BAR_WIDTH,
) -> None:
    """Draw mirrored solid bars with Sn upward and Bi downward."""
    if waves.size == 0:
        return

    axis.bar(
        waves,
        counts * direction,
        width=width,
        color=color,
        alpha=MIRROR_BAR_ALPHA,
        edgecolor="none",
        align="center",
        zorder=3,
    )


def map_counts_to_curve(
    selected_waves: np.ndarray,
    selected_counts: np.ndarray,
    spectrum_waves: np.ndarray,
    spectrum_values: np.ndarray,
    tolerance: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map selected bands and their frequencies to the nearest spectrum points."""
    curve_x: list[float] = []
    curve_y: list[float] = []
    curve_counts: list[float] = []

    for target, count in zip(selected_waves, selected_counts):
        idx = int(np.abs(spectrum_waves - target).argmin())
        if abs(float(spectrum_waves[idx]) - float(target)) <= tolerance:
            curve_x.append(float(spectrum_waves[idx]))
            curve_y.append(float(spectrum_values[idx]))
            curve_counts.append(float(count))

    return np.array(curve_x), np.array(curve_y), np.array(curve_counts)


def curve_marker_size(counts: np.ndarray) -> np.ndarray:
    """Scale curve markers by selection frequency."""
    return 18 + counts * 14


def semicircle_marker(side: str) -> mpath.Path:
    """Return a filled left or right half-circle marker."""
    if side == "left":
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 32)
    else:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 32)

    vertices = [(0.0, 0.0)] + [(np.cos(t), np.sin(t)) for t in theta] + [(0.0, 0.0)]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * len(theta) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(vertices, codes)


def make_wavenumber_unit_superscript(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = text.replace(
        "Wavenumbers (cm-1)",
        'Wavenumbers (cm<tspan baseline-shift="super" font-size="70%">-1</tspan>)',
    )
    svg_path.write_text(text, encoding="utf-8")


wavelengths, mean_spectrum = load_mean_spectrum(DATA_FILE)
sn_waves, sn_counts = summarize_frequency(SELECTION_RECORDS, "Sn")
bi_waves, bi_counts = summarize_frequency(SELECTION_RECORDS, "Bi")


# ==========================================
# 3. Figure
# ==========================================
fig = plt.figure(figsize=FIG_SIZE)
grid = fig.add_gridspec(
    nrows=2,
    ncols=1,
    height_ratios=(4.0, FREQ_PANEL_HEIGHT_RATIO),
    hspace=0.03,
)
ax_spectrum = fig.add_subplot(grid[0])
ax_freq = fig.add_subplot(grid[1], sharex=ax_spectrum)

for axis in (ax_spectrum, ax_freq):
    for zone in ZONES:
        start, end = zone["range"]
        axis.axvspan(start, end, facecolor=zone["color"], alpha=0.34, lw=0, zorder=0)

ax_spectrum.plot(wavelengths, mean_spectrum, color=MEAN_COLOR, lw=1.6, zorder=2)

ax_freq.axhline(0, color=BASELINE_COLOR, lw=1.0, alpha=0.9, zorder=1)
draw_mirrored_frequency_bars(ax_freq, sn_waves, sn_counts, direction=1.0, color=SN_COLOR)
draw_mirrored_frequency_bars(ax_freq, bi_waves, bi_counts, direction=-1.0, color=BI_COLOR)

sn_count_map = {float(wave): float(count) for wave, count in zip(sn_waves, sn_counts)}
bi_count_map = {float(wave): float(count) for wave, count in zip(bi_waves, bi_counts)}
shared_waves = np.array(sorted(set(sn_waves).intersection(set(bi_waves))), dtype=float)
sn_only_waves = np.array(sorted(set(sn_waves) - set(shared_waves)), dtype=float)
bi_only_waves = np.array(sorted(set(bi_waves) - set(shared_waves)), dtype=float)

sn_only_counts = np.array([sn_count_map[float(wave)] for wave in sn_only_waves], dtype=float)
bi_only_counts = np.array([bi_count_map[float(wave)] for wave in bi_only_waves], dtype=float)
shared_counts = np.array([
    max(sn_count_map[float(wave)], bi_count_map[float(wave)])
    for wave in shared_waves
], dtype=float)

sn_curve_x, sn_curve_y, sn_curve_counts = map_counts_to_curve(
    sn_only_waves, sn_only_counts, wavelengths, mean_spectrum
)
bi_curve_x, bi_curve_y, bi_curve_counts = map_counts_to_curve(
    bi_only_waves, bi_only_counts, wavelengths, mean_spectrum
)
shared_curve_x, shared_curve_y, shared_curve_counts = map_counts_to_curve(
    shared_waves, shared_counts, wavelengths, mean_spectrum
)

if sn_curve_x.size:
    ax_spectrum.scatter(sn_curve_x, sn_curve_y, s=curve_marker_size(sn_curve_counts), color=SN_COLOR, edgecolor="white",
                        linewidth=0.55, marker="o", zorder=5, clip_on=False)
if bi_curve_x.size:
    ax_spectrum.scatter(bi_curve_x, bi_curve_y, s=curve_marker_size(bi_curve_counts), color=BI_COLOR, edgecolor="white",
                        linewidth=0.55, marker="o", zorder=5, clip_on=False)
if shared_curve_x.size:
    shared_sizes = curve_marker_size(shared_curve_counts)
    ax_spectrum.scatter(shared_curve_x, shared_curve_y, s=shared_sizes, color=SN_COLOR,
                        marker=semicircle_marker("left"), linewidth=0, zorder=6, clip_on=False)
    ax_spectrum.scatter(shared_curve_x, shared_curve_y, s=shared_sizes, color=BI_COLOR,
                        marker=semicircle_marker("right"), linewidth=0, zorder=6, clip_on=False)
    ax_spectrum.scatter(shared_curve_x, shared_curve_y, s=shared_sizes, facecolors="none",
                        edgecolors="white", linewidth=0.55, marker="o", zorder=7, clip_on=False)


# ==========================================
# 4. Styling
# ==========================================
ax_spectrum.set_xlim(390, 1810)
ax_spectrum.set_ylabel("Intensity", fontsize=FONT_SIZE_LABEL)
ax_spectrum.tick_params(axis="x", labelbottom=False)

y_min = float(np.nanmin(mean_spectrum))
y_range = float(np.nanmax(mean_spectrum)) - y_min
bottom_margin = y_range * 0.10 if y_range > 0 else 0.1
ax_spectrum.set_ylim(y_min - bottom_margin, 3000)

ax_freq.set_ylim(-(TOTAL_FOLDS + 0.55), TOTAL_FOLDS + 0.55)
ax_freq.set_yticks([-5, -3, -1, 0, 1, 3, 5])
ax_freq.set_yticklabels(["5", "3", "1", "", "1", "3", "5"], fontsize=FONT_SIZE_TICK)
ax_freq.set_ylabel("Frequency", fontsize=FONT_SIZE_SECONDARY_LABEL)
ax_freq.set_xlabel("Wavenumbers (cm-1)", fontsize=FONT_SIZE_LABEL)
ax_freq.tick_params(axis="y", length=5)
ax_freq.text(0.035, 3.1, "Sn", transform=ax_freq.get_yaxis_transform(),
             color="black", fontsize=FONT_SIZE_TICK, va="center", ha="left")
ax_freq.text(0.035, -3.1, "Bi", transform=ax_freq.get_yaxis_transform(),
             color="black", fontsize=FONT_SIZE_TICK, va="center", ha="left")

for axis in (ax_spectrum, ax_freq):
    axis.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK, direction="in", length=5, width=1.2)
    axis.tick_params(axis="both", which="minor", direction="in", length=3, width=0.7)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)

ax_spectrum.yaxis.set_minor_locator(AutoMinorLocator(2))

legend_handles = [
    Line2D([0], [0], color=MEAN_COLOR, lw=1.6, label="Mean spectrum"),
    Patch(facecolor=SN_COLOR, edgecolor="none", alpha=MIRROR_BAR_ALPHA, label="Sn-selected bands"),
    Patch(facecolor=BI_COLOR, edgecolor="none", alpha=MIRROR_BAR_ALPHA, label="Bi-selected bands"),
]
ax_spectrum.legend(
    handles=legend_handles,
    loc="upper right",
    ncol=1,
    frameon=False,
    fontsize=FONT_SIZE_LEGEND,
    handlelength=1.6,
    columnspacing=0.8,
    labelspacing=0.45,
)

fig.align_ylabels([ax_spectrum, ax_freq])
fig.savefig(OUTPUT_FILE, format="svg", dpi=300, bbox_inches="tight")
make_wavenumber_unit_superscript(OUTPUT_FILE)
print(f"Figure saved: {OUTPUT_FILE}")

if "agg" not in matplotlib.get_backend().lower():
    plt.show()
