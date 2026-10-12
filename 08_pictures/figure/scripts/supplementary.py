"""Compose Figure S1 in the original 19-row, 10-column sample layout."""
from __future__ import annotations
import os
import re
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from publication_style import PALETTE

POLYMERS = ['POM','PE','PVC','SMMA','PAR','PPO','EPS','PTFE','ABS','TPU',
            'SD','LDPE','POE','SMAC','PC','PA6','PBT','PPA','MABS']


def draw_si(raw, inventory):
    numeric = []
    for col in raw.columns:
        try:
            float(col)
        except (ValueError, TypeError):
            continue
        numeric.append(col)
    waves = np.asarray(numeric, dtype=float)
    order = np.argsort(waves)
    waves = waves[order]
    values = raw[numeric].to_numpy(dtype=float)[:, order]
    assert np.isfinite(values).all()
    grouped = {name: [] for name in POLYMERS}
    for name, y in zip(raw.iloc[:, 0].astype(str), values):
        sample = int(re.search(r'\d+', name).group())
        polymer = inventory[sample]
        grouped[polymer].append((sample, name, y))
    assert sum(map(len, grouped.values())) == len(raw) == 187
    assert all(0 < len(v) <= 10 for v in grouped.values())
    # Preserve the original full-plate aspect ratio while keeping every glyph >= 9 pt.
    width_mm = 550
    height_mm = width_mm * 18223 / 11209
    fig, axes = plt.subplots(19, 10, figsize=(width_mm/25.4, height_mm/25.4))
    fig.subplots_adjust(left=.065, right=.992, bottom=.025, top=.983,
                        wspace=.55, hspace=.92)
    unused = []
    for row, polymer in enumerate(POLYMERS):
        samples = sorted(grouped[polymer])
        for col, ax in enumerate(axes[row]):
            if col >= len(samples):
                ax.set_visible(False)
                unused.append(ax)
                continue
            _, name, y = samples[col]
            ax.set_gid(name)
            ax.plot(waves, y, color=PALETTE['blue_dark'], lw=.55)
            ax.set_xlim(waves.min(), waves.max())
            ax.set_title(name, loc='left', fontsize=9, pad=4)
            ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11, labelpad=2)
            ax.set_ylabel('Intensity (a.u.)', fontsize=11, labelpad=2)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(direction='out', length=2, width=.8, pad=2)
        first = axes[row, 0].get_position()
        fig.text(.008, (first.y0+first.y1)/2, polymer, fontsize=11,
                 fontweight='bold', ha='left', va='center')
    # Missing last cells are intentional; compare gutters within contiguous rows.
    column_groups = []
    for col in range(10):
        group = []
        for row in range(19):
            ax = axes[row, col]
            if ax.get_visible():
                group.append(ax.get_gid())
            else:
                if len(group) > 1:
                    column_groups.append(group)
                group = []
        if len(group) > 1:
            column_groups.append(group)
    fig._alignment_options = {'exclude_axes': unused, 'column_groups': column_groups}
    return fig
