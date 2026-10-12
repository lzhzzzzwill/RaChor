from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import path as mpath
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from selection_records import folds, selected_wavelengths


# ==========================================
# 1. Plot configuration
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "original.xlsx"
FOLD_FILE = SCRIPT_DIR / "fold.xlsx"
OUTPUT_TEMPLATE = "mean_spectrum_selected_bands_fold{fold}.svg"

FONT_FAMILY = "Arial"
FIG_SIZE = (7.5, 6)
FONT_SIZE_LABEL = 30
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 17

SN_COLOR = "#C73E3A"
BI_COLOR = "#2F6FB0"
MEAN_COLOR = "#202020"
ZONE_ALPHA = 0.34

ZONES = [
    {"name": "LFR", "range": (400, 900), "color": "#D8E7F5"},
    {"name": "MFR", "range": (900, 1400), "color": "#DDEDD8"},
    {"name": "HFR", "range": (1400, 1800), "color": "#FFF1C7"},
]

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})


# ==========================================
# 2. Data loading
# ==========================================
def load_training_spectra(
    spectra_path: Path,
    fold_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Return all-sample mean and fold-specific training mean spectra."""
    try:
        spectra_df = pd.read_excel(spectra_path, header=0)
        fold_df = pd.read_excel(fold_path, header=0)

        sample_col = spectra_df.columns[0]
        spectra_df[sample_col] = spectra_df[sample_col].astype(str).str.strip()
        fold_df["Sample"] = fold_df["Sample"].astype(str).str.strip()

        numeric_columns = pd.to_numeric(spectra_df.columns[1:], errors="coerce")
        spectral_columns = [
            col for col, wave in zip(spectra_df.columns[1:], numeric_columns)
            if pd.notna(wave) and 400 <= float(wave) <= 1800
        ]
        wavelengths = np.array([float(col) for col in spectral_columns], dtype=float)

        merged_df = spectra_df[[sample_col, *spectral_columns]].merge(
            fold_df[["Sample", "OuterFold"]],
            left_on=sample_col,
            right_on="Sample",
            how="inner",
        )
        if merged_df.empty:
            raise ValueError("No samples matched between original.xlsx and fold.xlsx.")

        mean_spectrum = merged_df[spectral_columns].mean(axis=0).to_numpy()
        training_spectra: dict[int, np.ndarray] = {}
        for fold in sorted(merged_df["OuterFold"].dropna().unique()):
            fold_id = int(fold)
            train_df = merged_df[merged_df["OuterFold"] != fold]
            training_spectra[fold_id] = train_df[spectral_columns].mean(axis=0).to_numpy()

        print(
            f"Data loaded: {merged_df.shape[0]} matched samples, "
            f"{len(spectral_columns)} bands, {len(training_spectra)} folds"
        )
        return wavelengths, mean_spectrum, training_spectra
    except Exception as exc:
        print(f"Failed to read fold-specific spectra. Demo data are used instead. ({exc})")
        wavelengths = np.linspace(400, 1800, 700)
        mean_spectrum = (
            1.00 * np.exp(-((wavelengths - 550) ** 2) / (2 * 60 ** 2))
            + 0.80 * np.exp(-((wavelengths - 850) ** 2) / (2 * 80 ** 2))
            + 0.60 * np.exp(-((wavelengths - 1250) ** 2) / (2 * 70 ** 2))
            + 0.30
        )
        training_spectra = {
            fold: mean_spectrum * (1 + (fold - 3) * 0.008)
            for fold in range(1, 6)
        }
        return wavelengths, mean_spectrum, training_spectra


def match_available_bands(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    targets: list[float],
    tolerance: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map target wavelengths to the nearest available measured bands."""
    selected_x: list[float] = []
    selected_y: list[float] = []

    for target in targets:
        idx = int(np.abs(wavelengths - target).argmin())
        if abs(float(wavelengths[idx]) - target) <= tolerance:
            selected_x.append(float(wavelengths[idx]))
            selected_y.append(float(spectrum[idx]))

    return np.array(selected_x), np.array(selected_y)


def assign_offsets(values: np.ndarray, min_spacing: float = 22.0) -> np.ndarray:
    """Stagger dense selected bands within the same feature row."""
    if values.size == 0:
        return values

    order = np.argsort(values)
    levels = np.zeros(values.size, dtype=int)
    last_at_level = np.full(5, -np.inf)

    for idx in order:
        for level in range(len(last_at_level)):
            if values[idx] - last_at_level[level] >= min_spacing:
                levels[idx] = level
                last_at_level[level] = values[idx]
                break

    offsets = np.array([0.00, 0.17, -0.17, 0.34, -0.34])
    return offsets[levels]


def semicircle_marker(side: str) -> mpath.Path:
    """Return a filled left or right half-circle marker."""
    if side == "left":
        theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 32)
    else:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 32)

    vertices = [(0.0, 0.0)] + [(np.cos(t), np.sin(t)) for t in theta] + [(0.0, 0.0)]
    codes = [mpath.Path.MOVETO] + [mpath.Path.LINETO] * len(theta) + [mpath.Path.CLOSEPOLY]
    return mpath.Path(vertices, codes)


def split_selection_points(
    sn_points: tuple[np.ndarray, np.ndarray],
    bi_points: tuple[np.ndarray, np.ndarray],
    tolerance: float = 1e-6,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split selected bands into Sn-only, Bi-only, and shared points."""
    sn_values, sn_intensities = sn_points
    bi_values, bi_intensities = bi_points

    sn_only_x, sn_only_y = [], []
    bi_only_x, bi_only_y = [], []
    shared_x, shared_y = [], []
    matched_bi_indices: set[int] = set()

    for x_value, y_value in zip(sn_values, sn_intensities):
        matches = np.where(np.isclose(bi_values, x_value, atol=tolerance))[0]
        if matches.size:
            shared_x.append(float(x_value))
            shared_y.append(float(y_value))
            matched_bi_indices.update(int(i) for i in matches)
        else:
            sn_only_x.append(float(x_value))
            sn_only_y.append(float(y_value))

    for idx, (x_value, y_value) in enumerate(zip(bi_values, bi_intensities)):
        if idx not in matched_bi_indices:
            bi_only_x.append(float(x_value))
            bi_only_y.append(float(y_value))

    return {
        "sn_only": (np.array(sn_only_x), np.array(sn_only_y)),
        "bi_only": (np.array(bi_only_x), np.array(bi_only_y)),
        "shared": (np.array(shared_x), np.array(shared_y)),
    }


def make_wavenumber_unit_superscript(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = text.replace(
        "Wavenumbers (cm-1)",
        'Wavenumbers (cm<tspan baseline-shift="super" font-size="70%">-1</tspan>)',
    )
    svg_path.write_text(text, encoding="utf-8")


# ==========================================
# 3. Figure
# ==========================================
def plot_fold(
    fold: int,
    wavelengths: np.ndarray,
    training_spectrum: np.ndarray,
    output_path: Path,
) -> None:
    sn_targets = selected_wavelengths(fold, "Sn")
    bi_targets = selected_wavelengths(fold, "Bi")
    sn_x, sn_y = match_available_bands(wavelengths, training_spectrum, sn_targets)
    bi_x, bi_y = match_available_bands(wavelengths, training_spectrum, bi_targets)
    curve_points = split_selection_points((sn_x, sn_y), (bi_x, bi_y))

    fig = plt.figure(figsize=FIG_SIZE)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=(4.0, 1.05),
        hspace=0.03,
    )
    ax = fig.add_subplot(grid[0])
    ax_band = fig.add_subplot(grid[1], sharex=ax)

    for axis in (ax, ax_band):
        for zone in ZONES:
            start, end = zone["range"]
            axis.axvspan(start, end, facecolor=zone["color"], alpha=ZONE_ALPHA, lw=0, zorder=0)

    ax.plot(wavelengths, training_spectrum, color=MEAN_COLOR, lw=1.6, zorder=2)

    sn_only_x, sn_only_y = curve_points["sn_only"]
    bi_only_x, bi_only_y = curve_points["bi_only"]
    shared_x, shared_y = curve_points["shared"]

    if sn_only_x.size:
        ax.scatter(sn_only_x, sn_only_y, s=28, color=SN_COLOR, edgecolor="white",
                   linewidth=0.5, marker="o", zorder=4, clip_on=False)
    if bi_only_x.size:
        ax.scatter(bi_only_x, bi_only_y, s=28, color=BI_COLOR, edgecolor="white",
                   linewidth=0.5, marker="o", zorder=4, clip_on=False)
    if shared_x.size:
        ax.scatter(shared_x, shared_y, s=42, color=SN_COLOR, marker=semicircle_marker("left"),
                   linewidth=0, zorder=5, clip_on=False)
        ax.scatter(shared_x, shared_y, s=42, color=BI_COLOR, marker=semicircle_marker("right"),
                   linewidth=0, zorder=5, clip_on=False)
        ax.scatter(shared_x, shared_y, s=42, facecolors="none", edgecolors="white",
                   linewidth=0.5, marker="o", zorder=6, clip_on=False)

    for selected_x, color in ((sn_x, SN_COLOR), (bi_x, BI_COLOR)):
        if selected_x.size:
            ax_band.vlines(selected_x, ymin=-0.55, ymax=1.55, color=color, lw=0.45, alpha=0.28, zorder=1)

    ax_band.scatter(
        sn_x,
        np.full_like(sn_x, 1.0) + assign_offsets(sn_x),
        s=34,
        color=SN_COLOR,
        marker="o",
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax_band.scatter(
        bi_x,
        np.full_like(bi_x, 0.0) + assign_offsets(bi_x),
        s=38,
        color=BI_COLOR,
        marker="o",
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax_band.hlines([0, 1], xmin=400, xmax=1800, color="#777777", lw=0.6, alpha=0.55, zorder=2)

    ax.set_xlim(390, 1810)
    ax.set_ylabel("Intensity", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK, direction="in", length=5, width=1.2)
    ax.tick_params(axis="both", which="minor", direction="in", length=3, width=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    y_min = float(np.nanmin(training_spectrum))
    y_range = float(np.nanmax(training_spectrum)) - y_min
    y_margin = y_range * 0.10 if y_range > 0 else 0.1
    ax.set_ylim(y_min - y_margin, 3000)

    ax_band.set_ylim(-0.86, 1.38)
    ax_band.set_yticks([0, 1])
    ax_band.set_yticklabels(["Bi", "Sn"], fontsize=FONT_SIZE_TICK)
    ax_band.set_xlabel("Wavenumbers (cm-1)", fontsize=FONT_SIZE_LABEL)
    ax_band.tick_params(axis="x", which="major", labelsize=FONT_SIZE_TICK, direction="in", length=5, width=1.2)
    ax_band.tick_params(axis="x", which="minor", direction="in", length=3, width=0.7)
    ax_band.tick_params(axis="y", length=0)
    ax_band.xaxis.set_minor_locator(AutoMinorLocator(2))

    for axis in (ax, ax_band):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.2)
        axis.spines["bottom"].set_linewidth(1.2)

    legend_handles = [
        Line2D([0], [0], color=MEAN_COLOR, lw=1.6, label=f"Fold {fold} training mean"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SN_COLOR,
               markeredgecolor="white", markersize=7, label="Sn selected bands"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BI_COLOR,
               markeredgecolor="white", markersize=7, label="Bi selected bands"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=FONT_SIZE_LEGEND, handlelength=2.4)

    fig.align_ylabels([ax, ax_band])
    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    make_wavenumber_unit_superscript(output_path)
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main() -> None:
    wavelengths, _, fold_training_spectra = load_training_spectra(DATA_FILE, FOLD_FILE)
    for fold in folds():
        if fold not in fold_training_spectra:
            available_folds = ", ".join(str(item) for item in sorted(fold_training_spectra))
            raise ValueError(f"Fold {fold} is not available. Available folds: {available_folds}")

        output_path = SCRIPT_DIR / OUTPUT_TEMPLATE.format(fold=fold)
        plot_fold(fold, wavelengths, fold_training_spectra[fold], output_path)

    if "agg" not in matplotlib.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
