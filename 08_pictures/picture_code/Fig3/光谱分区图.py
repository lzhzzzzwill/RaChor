from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt


# ==========================================
# 1. Plot configuration
# ==========================================
FONT_FAMILY = "Arial"
FIG_SIZE = (9.5, 1.0)
FONT_SIZE_LABEL = 30
FONT_SIZE_RANGE = 22
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR / "spectral_zones.svg"
ZONE_ALPHA = 0.34

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.sans-serif": [FONT_FAMILY],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})


def make_zone_units_superscript(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(
        r"(\d+-\d+) cm-1",
        r'\1 cm<tspan baseline-shift="super" font-size="70%">-1</tspan>',
        text,
    )
    svg_path.write_text(text, encoding="utf-8")


# ==========================================
# 2. Spectral zones
# ==========================================
ZONES = [
    {"name": "LFR", "range": (400, 900), "color": "#D8E7F5"},
    {"name": "MFR", "range": (900, 1400), "color": "#DDEDD8"},
    {"name": "HFR", "range": (1400, 1800), "color": "#FFF1C7"},
]

# ==========================================
# 3. Figure
# ==========================================
fig, ax = plt.subplots(figsize=FIG_SIZE)

for zone in ZONES:
    start, end = zone["range"]
    width = end - start
    center_x = start + width / 2

    ax.barh(
        y=0,
        width=width,
        left=start,
        height=0.42,
        color=zone["color"],
        alpha=ZONE_ALPHA,
        edgecolor="none",
        align="center",
    )

    ax.text(
        center_x,
        0.09,
        zone["name"],
        ha="center",
        va="center",
        fontsize=FONT_SIZE_LABEL,
        fontfamily=FONT_FAMILY,
        color="black",
    )
    ax.text(
        center_x,
        -0.16,
        f"{start}-{end} cm-1",
        ha="center",
        va="center",
        fontsize=FONT_SIZE_RANGE,
        fontfamily=FONT_FAMILY,
        color="black",
    )


# ==========================================
# 4. Styling
# ==========================================
ax.set_xlim(ZONES[0]["range"][0], ZONES[-1]["range"][1])
ax.axis("off")

plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
plt.savefig(OUTPUT_FILE, format="svg", dpi=300, bbox_inches="tight", pad_inches=0.01)
make_zone_units_superscript(OUTPUT_FILE)
print(f"Figure saved: {OUTPUT_FILE}")
if "agg" not in matplotlib.get_backend().lower():
    plt.show()
