"""Shared publication styling and export helpers for performance figures."""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment

PALETTE = {
    'lightest': '#F7FCF0', 'light': '#E0F3DB', 'green_light': '#CCEBC5',
    'green': '#A8DDB5', 'cyan': '#7BCCC4', 'blue_light': '#4EB3D3',
    'blue': '#2B8CBE', 'blue_dark': '#0868AC', 'darkest': '#084081',
}

def apply_nature_style(font_size: float = 9.0) -> None:
    mpl.rcParams.update({
        # Explicit second family supplies superscript minus absent from Arial.
        'font.family': ['sans-serif', 'DejaVu Sans'],
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': font_size,
        'axes.labelsize': font_size + 2, 'axes.titlesize': font_size,
        'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'legend.fontsize': font_size, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })

def project_paths() -> tuple[Path, Path]:
    vis_dir = Path(__file__).resolve().parent.parent
    return vis_dir / 'data', vis_dir / 'output' / 'performance'

def ensure_columns(frame, required: list[str], source: Path) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f'{source.name} is missing required columns: {missing}')

def save_publication_figure(fig: plt.Figure, output_prefix: Path) -> list[Path]:
    """Export only a standalone PDF; keep layout diagnostics in temporary storage."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    options = getattr(fig, '_alignment_options', {})
    qa_dir = Path(__file__).resolve().parent.parent / '.cache' / 'alignment'
    qa_dir.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig,
        json_out=str(qa_dir / (output_prefix.name + '.alignment.json')),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **options)
    pdf_path = output_prefix.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    return [pdf_path]
