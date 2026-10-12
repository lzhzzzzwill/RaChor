"""Four source-grounded alternatives for Figure 2a and 2b; PDF exports only."""
from __future__ import annotations
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from publication_style import PALETTE, apply_nature_style, ensure_columns, project_paths, save_publication_figure

MM = 1/25.4
INK = PALETTE['darkest']
TARGETS = [('118Sn (KED)', 'Sn', PALETTE['blue_dark']),
           ('209Bi (KED)', 'Bi', PALETTE['blue_light'])]
SCHEMES = [('01_histogram_rug', 'Histogram + rug'),
           ('02_boxplot_points', 'Boxplot + observations'),
           ('03_raincloud', 'Raincloud'),
           ('04_ecdf', 'Empirical cumulative distribution')]


def point_lanes(values, half_width=0.09):
    """Deterministic visual offsets; preserve all observations and their x values."""
    rank = np.argsort(np.argsort(values, kind='stable'), kind='stable')
    return (np.mod(rank * 0.618033988749895, 1)-0.5)*2*half_width


def box(ax, y, position, color, width=.14):
    bp = ax.boxplot([y], vert=False, positions=[position], widths=width,
                    patch_artist=True, manage_ticks=False, showfliers=False, whis=1.5,
                    boxprops={'facecolor': color, 'alpha':.4, 'edgecolor':INK, 'linewidth':.8},
                    medianprops={'color':INK, 'linewidth':1.2},
                    whiskerprops={'color':INK, 'linewidth':.8},
                    capprops={'color':INK, 'linewidth':.8})
    return bp


def panel(ax, y, target, color, mode, letter, title=True):
    if title:
        ax.set_title(f'{target} response · n = {len(y)}', loc='left', pad=9)
    ax.annotate(f'({letter})', xy=(0,1), xycoords='axes fraction', xytext=(-30,16),
                textcoords='offset points', fontsize=12, fontweight='bold', va='bottom',
                annotation_clip=False)
    ax.set_xlim(0,210)
    ax.set_xticks([0,50,100,150,200])
    ax.set_xlabel('Measured response')
    ax.tick_params(direction='out', length=3, pad=6)
    if mode == 'histogram_rug':
        # Shared 10-response-unit bins, inclusive of every observed value.
        bins = np.arange(0,211,10)
        counts, _ = np.histogram(y,bins=bins)
        assert counts.sum() == len(y)
        ax.hist(y,bins=bins,color=color,alpha=.78,edgecolor='white',linewidth=.55)
        ax.plot(y,np.full(len(y),-.025), '|', color=INK,markersize=3.5,
                markeredgewidth=.45,transform=ax.get_xaxis_transform(),clip_on=False)
        ax.set_ylabel('Sample count')
        ax.set_ylim(0, max(counts)*1.13)
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    elif mode == 'boxplot_points':
        box(ax,y,.62,color)
        ax.scatter(y,.22+point_lanes(y),s=8,color=color,alpha=.65,
                   edgecolor=INK,linewidth=.15)
        ax.set_ylim(0,.90)
        ax.set_yticks([.22,.62],['Samples','Boxplot'])
        ax.tick_params(axis='y',length=0,pad=5)
        ax.spines['left'].set_visible(False)
    elif mode == 'raincloud':
        kde = gaussian_kde(y,bw_method='scott')
        grid = np.linspace(0,210,700)
        density = kde(grid)
        shape = .42+.48*density/density.max()
        ax.fill_between(grid,.42,shape,color=color,alpha=.4,lw=0)
        ax.plot(grid,shape,color=color,lw=.9)
        box(ax,y,.28,color,width=.10)
        ax.scatter(y,.10+point_lanes(y,.048),s=7,color=color,alpha=.65,
                   edgecolor=INK,linewidth=.15)
        ax.set_ylim(0,1)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    elif mode == 'ecdf':
        ordered=np.sort(y)
        unique,counts=np.unique(ordered,return_counts=True)
        cumulative=np.cumsum(counts)/len(y)
        ax.step(np.r_[0,unique,210],np.r_[0,cumulative,1],where='post',color=color,lw=1.2)
        ax.plot(unique,cumulative,ls='',marker='o',markersize=1.8,color=color)
        ax.set_ylabel('Cumulative proportion')
        ax.set_ylim(0,1.03)
        ax.set_yticks([0,.25,.5,.75,1])
    else:
        raise ValueError(mode)


def main():
    apply_nature_style()
    data,_=project_paths()
    source=data/'processed_model.csv'
    frame=pd.read_csv(source)
    ensure_columns(frame,[t[0] for t in TARGETS],source)
    observations=[frame[col].to_numpy(dtype=float) for col,_,_ in TARGETS]
    assert len(frame)==187
    assert all(np.isfinite(y).all() and y.min()>=0 and y.max()<=210 for y in observations)
    output=data.parent/'output/response_options'
    for name,heading in SCHEMES:
        mode=name[3:]
        fig,axes=plt.subplots(1,2,figsize=(183*MM,72*MM))
        fig.subplots_adjust(left=.095,right=.985,bottom=.26,top=.82,wspace=.42)
        for ax,y,(_,target,color),letter in zip(axes,observations,TARGETS,'ab'):
            panel(ax,y,target,color,mode,letter)
        if mode=='histogram_rug':
            ceiling=max(np.histogram(y,bins=np.arange(0,211,10))[0].max() for y in observations)*1.13
            for ax in axes:ax.set_ylim(0,ceiling)
        note = {'histogram_rug':'Bin width: 10 response units; rug marks include all 187 samples per target.',
                'boxplot_points':'Box: median and IQR; whiskers: 1.5×IQR; points include all observed samples.',
                'raincloud':'KDE: Scott bandwidth (smoothed); box: median/IQR, 1.5×IQR whiskers; all samples shown.',
                'ecdf':'Each step represents the observed fraction; all 187 samples per target are included.'}[mode]
        fig.text(.095,.03,note,fontsize=9,ha='left')
        save_publication_figure(fig,output/name)
        plt.close(fig)
        print(name,flush=True)
    fig,axes=plt.subplots(4,2,figsize=(183*MM,245*MM))
    fig.subplots_adjust(left=.105,right=.985,bottom=.105,top=.91,hspace=1.18,wspace=.45)
    for row,(name,heading) in enumerate(SCHEMES):
        for col,(y,(_,target,color)) in enumerate(zip(observations,TARGETS)):
            panel(axes[row,col],y,target,color,name[3:],chr(97+row*2+col),title=False)
        if row==0:
            ceiling=max(np.histogram(y,bins=np.arange(0,211,10))[0].max() for y in observations)*1.13
            for ax in axes[row]:ax.set_ylim(0,ceiling)
        for ax in axes[row]:ax.set_title(heading,loc='left',pad=8)
    for col,(_,target,color) in enumerate(TARGETS):
        pos=axes[0,col].get_position()
        fig.text(pos.x0,.968,f'{target} response · n = 187',fontsize=11,fontweight='bold',color=INK)
    fig.text(.105,.016,'Same observations and response scale in every option; density is a descriptive smoothing.',fontsize=9)
    save_publication_figure(fig,output/'00_options_comparison')
    fig.savefig(Path(tempfile.gettempdir())/'response_options_comparison.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('00_options_comparison',flush=True)

if __name__=='__main__':main()
