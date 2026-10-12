"""Redraw the complete scientific figure set using measured source data only."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import string
import sys
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from response_options import panel as response_panel
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from publication_style import PALETTE as P, apply_nature_style, ensure_columns, project_paths, save_publication_figure

BASE = Path(__file__).resolve().parent.parent
PROJECT = BASE.parent.parent
SOURCE = PROJECT / '08_pictures' / 'picture_code'
DATA, _ = project_paths()
OUT = BASE / 'output'
MM = 1 / 25.4
SN, BI, INK = P['blue_dark'], P['cyan'], P['darkest']
WAVE_LABEL = 'Wavenumber (cm⁻¹)'
MANIFEST = []
METRICS = []
NOTES = []


def prepare_data(refresh=False):
    snapshots = ['raw_spectra.xlsx', 'oof_predictions.xlsx', 'folds.xlsx',
                 'feature_sweep.xlsx', 'processed_all.csv', 'processed_model.csv',
                 'performance.json', 'selection_records.json', 'top40_bands.json',
                 'polymer_inventory.json']
    if not refresh and all((DATA/name).is_file() for name in snapshots):
        return (json.loads((DATA/'performance.json').read_text()),
                json.loads((DATA/'selection_records.json').read_text()),
                {int(k): v for k, v in json.loads((DATA/'polymer_inventory.json').read_text()).items()})
    DATA.mkdir(parents=True, exist_ok=True)
    files = {
        SOURCE / 'original.xlsx': 'raw_spectra.xlsx',
        SOURCE / 'nestedcv_oof_predictions.xlsx': 'oof_predictions.xlsx',
        SOURCE / 'Fig3/fold.xlsx': 'folds.xlsx',
        SOURCE / 'Fig7/data.xlsx': 'feature_sweep.xlsx',
        SOURCE / 'Fig1/filter_snv_sg_scaled.csv': 'processed_all.csv',
        SOURCE / 'Fig2/filter_snv_sg_scaled_del_4_model.csv': 'processed_model.csv',
        PROJECT / '投稿/si/04_Supplementary_Data.xlsx': 'supplementary.xlsx',
    }
    records = []
    for src, name in files.items():
        dest = DATA / name
        if refresh or not dest.exists():
            shutil.copy2(src, dest)
        records.append({'file': name, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    # Extract only literal data assignments; never execute historical plotting code.
    module = ast.parse((SOURCE / '箱线图.py').read_text())
    performance = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('DATA_') for t in node.targets):
            expression = ast.unparse(node.value).replace('np.nan', 'None')
            performance[node.targets[0].id] = ast.literal_eval(expression)
    (DATA / 'performance.json').write_text(json.dumps(performance, indent=2))
    module = ast.parse((SOURCE / 'Fig3/selection_records.py').read_text())
    selections = next(ast.literal_eval(n.value) for n in module.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SELECTION_RECORDS' for t in n.targets))
    (DATA / 'selection_records.json').write_text(json.dumps(selections, indent=2))
    modules = ast.parse((SOURCE / 'Fig1/TOP40波段示意图.py').read_text())
    top40 = next(ast.literal_eval(n.value) for n in modules.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'selected_waves' for t in n.targets))
    (DATA / 'top40_bands.json').write_text(json.dumps(top40))
    # Existing appendix files provide the published polymer assignment; do not reuse the obsolete range guess.
    inventory = {}
    for src in (SOURCE / 'Spectra_Plots').glob('*/*.svg'):
        inventory[int(src.stem)] = src.parent.name
    (DATA / 'polymer_inventory.json').write_text(json.dumps(inventory, indent=2))
    (DATA / 'source_checksums.json').write_text(json.dumps(records, indent=2))
    return performance, selections, inventory


def spectra(frame, low=None, high=None):
    cols = []
    for c in frame.columns:
        try: w = float(c)
        except (ValueError, TypeError): continue
        if (low is None or w >= low) and (high is None or w <= high): cols.append(c)
    waves = np.asarray(cols, dtype=float)
    vals = frame[cols].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): raise ValueError('Nonfinite measured spectral values; no silent imputation allowed')
    order = np.argsort(waves)
    return waves[order], vals[:, order]


def new_single(width=89, height=65):
    fig, ax = plt.subplots(figsize=(width*MM, height*MM))
    fig.subplots_adjust(left=.17, right=.98, bottom=.20, top=.86)
    return fig, ax


def label(ax, letter):
    ax.annotate(f'({letter})', xy=(0, 1), xycoords='axes fraction', xytext=(-32, 16),
        textcoords='offset points', weight='bold', fontsize=12, va='bottom', ha='left', annotation_clip=False)


def axes_style(ax, xlabel=None, ylabel=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.tick_params(direction='out', length=3, width=.8, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def export(fig, name, claim, preview=False):
    path = OUT / name
    save_publication_figure(fig, path)
    plt.close(fig)
    print(name, flush=True)


def plot_prediction(ax, frame, target, model, title=True):
    metal = '118Sn (KED)' if target == 'Sn' else '209Bi (KED)'
    ensure_columns(frame, [metal+'_true', metal+'_pred'], DATA/'oof_predictions.xlsx')
    true, pred = frame[metal+'_true'].to_numpy(), frame[metal+'_pred'].to_numpy()
    assert np.isfinite(true).all() and np.isfinite(pred).all()
    rmse = np.sqrt(np.mean((true-pred)**2))
    r2 = 1 - np.sum((true-pred)**2)/np.sum((true-true.mean())**2)
    rpd = true.std(ddof=0) / rmse
    lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
    span = hi-lo
    lims = (lo-.08*span, hi+.08*span)
    identity, = ax.plot(lims, lims, ls='--', color='#777777', lw=.7, label='Y=T')
    slope, intercept = np.polyfit(true, pred, 1)
    fit, = ax.plot(lims, slope*np.array(lims)+intercept, color=INK, lw=.8, label='Fit')
    ax.scatter(true, pred, s=10, color=SN if model=='GlobalStack' else BI,
        edgecolor=INK, linewidth=.25, alpha=.78)
    ax.set_xlim(lims); ax.set_ylim(lims)
    axes_style(ax, 'Measured response', 'Predicted response')
    if title: ax.set_title(f'{target} · {"GlobalStack" if model=="GlobalStack" else "Final fusion"}', loc='left', pad=8)
    # Opposite empty corners hold metrics and reference-line labels.
    ax.text(.04, .96, f'$\\mathit{{R}}^{{2}}$: {r2:.4f}\n$\\mathit{{RMSE}}$: {rmse:.3f}\n$\\mathit{{RPD}}$: {rpd:.3f}', transform=ax.transAxes, fontsize=9, ha='left', va='top', linespacing=1.3)
    ax.legend(handles=[fit,identity],loc='lower right',handlelength=2,frameon=False)
    METRICS.append({'target':target,'model':model,'n':len(true),'R2':r2,'RMSE':rmse,'RPD_population_SD':rpd})


def plot_boxes(ax, values, target, subset):
    models = list(values)
    clean = [[float(v) for v in values[m] if v is not None] for m in models]
    boxes = ax.boxplot(clean, positions=np.arange(len(models))+1, widths=.58, patch_artist=True,
        whis=1.5, showfliers=False, medianprops={'color':INK,'linewidth':1},
        boxprops={'linewidth':.6,'edgecolor':INK}, whiskerprops={'linewidth':.6,'color':INK},
        capprops={'linewidth':.6,'color':INK})
    for i, box in enumerate(boxes['boxes']): box.set_facecolor(P['light'] if i < len(models)-2 else P['green'])
    markers = ['o','s','^','D','v']
    for i, model in enumerate(models):
        for fold, score in enumerate(values[model]):
            if score is None: continue
            ax.scatter(i+1+(fold-2)*.075, score, s=13, marker=markers[fold],
                facecolor=SN, edgecolor='white', linewidth=.25, zorder=3)
    ax.set_xticks(np.arange(len(models))+1)
    display = {'Global Stack':'GlobalStack','Final Fit':'Final fusion','LFR stack':'LFR','MFR stack':'MFR','HFR stack':'HFR'}
    ax.set_xticklabels([display.get(m,m) for m in models], rotation=90, ha='right', va='center', rotation_mode='anchor')
    ax.set_ylim(.6,1); ax.set_yticks([.6,.7,.8,.9,1])
    ax.set_ylabel('$\\mathit{R}^{2}$'); ax.set_title(f'{target} · {"Training CV" if subset=="train" else "Held-out folds"}',loc='left',pad=8)
    ax.tick_params(direction='out',length=3,pad=2)


def spectral_overview(ax, waves, values):
    mean = values.mean(axis=0); sd = values.std(axis=0,ddof=1)
    ax.fill_between(waves,values.min(axis=0),values.max(axis=0),color=P['green_light'],label='Sample range')
    ax.fill_between(waves,mean-sd,mean+sd,color=P['cyan'],alpha=.7,label='Mean ± 1 SD')
    ax.plot(waves,mean,color=SN,lw=.8,label='Mean')
    axes_style(ax,WAVE_LABEL,'Intensity (a.u.)'); ax.set_xlim(waves.min(),waves.max())
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=3,handlelength=1.1,columnspacing=.8)


def stages(values):
    raw = values[0]; spread=raw.std()
    if spread==0: raise ValueError('SNV undefined for constant spectrum')
    snv=(raw-raw.mean())/spread
    sg=savgol_filter(snv,9,2,deriv=1)
    std=(sg-sg.mean())/sg.std()
    return [raw,snv,sg,std]


def plot_sweep(ax,sweep,compact=False):
    ensure_columns(sweep,['SelectedMaxFeatures','Sn MultiTaskLassoCV','Bi MultiTaskLassoCV','Sn Lasso','Bi Lasso'],DATA/'feature_sweep.xlsx')
    x=sweep['SelectedMaxFeatures']
    for i,(column,metal,style,marker) in enumerate([
        ('Sn MultiTaskLassoCV','Sn','-','o'),('Bi MultiTaskLassoCV','Bi','-','s'),
        ('Sn Lasso','Sn','--','o'),('Bi Lasso','Bi','--','s')]):
        color=SN if metal=='Sn' else P['blue_light']
        y=sweep[column]
        ax.plot(x,y,color=color,ls=style,lw=1,marker=marker,markersize=2.3,markevery=10,label=column.replace('MultiTaskLassoCV','Shared selection'))
        idx=y.idxmax()
        ax.scatter(x.loc[idx],y.loc[idx],s=21,color=color,marker=marker,zorder=4,edgecolor='white',linewidth=.4)
    axes_style(ax,'Maximum selected features','OOF $\\mathit{R}^{2}$'); ax.set_xlim(x.min(),x.max())
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.35),ncol=2,columnspacing=1.5,handlelength=2)


def add_zones(ax):
    for (lo,hi,name,color) in [(400,900,'LFR',P['light']),(900,1400,'MFR',P['green_light']),(1400,1800,'HFR',P['green'])]:
        ax.axvspan(lo,hi,color=color,alpha=.6,lw=0,zorder=0)
    ax.set_xlim(400,1800)


def half_marker(left):
    from matplotlib.path import Path as MarkerPath
    theta=np.linspace(np.pi/2,3*np.pi/2,32) if left else np.linspace(-np.pi/2,np.pi/2,32)
    verts=[(0.,0.)]+list(zip(np.cos(theta),np.sin(theta)))+[(0.,0.)]
    return MarkerPath(verts,[MarkerPath.MOVETO]+[MarkerPath.LINETO]*32+[MarkerPath.CLOSEPOLY])


def fold_plot(ax,waves,values,records,fold):
    # Keep the original spectrum + target-specific selection-strip organization.
    nested = ax.get_subplotspec().subgridspec(2,1,height_ratios=[2.6,1.4],hspace=.26)
    ax.set_subplotspec(nested[0])
    ax.set_gid(f'fold{fold}_spectrum')
    bands = ax.figure.add_subplot(nested[1],sharex=ax)
    bands.set_gid(f'fold{fold}_bands')
    for axis in (ax,bands): add_zones(axis)
    mean=values.mean(axis=0)
    ax.plot(waves,mean,color=INK,lw=.8)
    chosen={}
    for target in ['Sn','Bi']:
        targets=next(r['wavelengths'] for r in records if r['fold']==fold and r['metal']==target)
        indices=np.array([np.abs(waves-w).argmin() for w in targets])
        if max(abs(waves[idx]-w) for idx,w in zip(indices,targets)) > 1:
            raise ValueError('Selected bands do not match measured wavenumbers')
        chosen[target]=indices
    shared=np.intersect1d(chosen['Sn'],chosen['Bi'])
    for target,color,row in [('Sn',SN,1),('Bi',P['blue_light'],0)]:
        indices=chosen[target]; unique=np.setdiff1d(indices,shared)
        ax.scatter(waves[unique],mean[unique],s=13,color=color,edgecolor='white',linewidth=.25,zorder=4)
        bands.vlines(waves[indices],-.45,1.45,color=color,lw=.4,alpha=.28)
        # Stagger nearby selections rather than concealing coincident markers.
        last=np.full(5,-np.inf);levels=np.zeros(len(indices),dtype=int)
        for j in np.argsort(waves[indices]):
            for level in range(5):
                if waves[indices[j]]-last[level]>=25:
                    levels[j]=level;last[level]=waves[indices[j]];break
        offsets=np.array([0,.17,-.17,.34,-.34])[levels]
        bands.scatter(waves[indices],row+offsets,s=14,color=color,edgecolor='white',linewidth=.25,zorder=3)
    for color,left in [(SN,True),(P['blue_light'],False)]:
        ax.scatter(waves[shared],mean[shared],s=18,color=color,marker=half_marker(left),linewidth=0,zorder=5)
    ax.scatter(waves[shared],mean[shared],s=18,facecolor='none',edgecolor='white',linewidth=.25,zorder=6)
    ax.set_title(f'Fold {fold} · training mean',loc='left',pad=8)
    axes_style(ax,None,'Intensity (a.u.)')
    ax.tick_params(axis='x',labelbottom=False,length=0)
    if mean.min()<0 or mean.max()>3000:
        raise ValueError('Mean spectrum exceeds the shared 0–3000 intensity scale')
    ax.set_ylim(0,3000);ax.set_yticks([0,1500,3000])
    bands.hlines([0,1],400,1800,color=INK,lw=.4,alpha=.3)
    axes_style(bands,WAVE_LABEL)
    bands.set_ylim(-.6,1.6);bands.set_yticks([1,0]);bands.set_yticklabels(['Sn','Bi'])
    bands.tick_params(axis='y',length=0,pad=3)
    return bands


def frequency_plot(ax,waves,values,records):
    from matplotlib.patches import Patch
    nested=ax.get_subplotspec().subgridspec(2,1,height_ratios=[2.6,1.4],hspace=.26)
    ax.set_subplotspec(nested[0]);ax.set_gid('frequency_spectrum')
    freq=ax.figure.add_subplot(nested[1],sharex=ax);freq.set_gid('frequency_bands')
    for axis in (ax,freq):add_zones(axis)
    mean=values.mean(axis=0)
    ax.plot(waves,mean,color=INK,lw=.8)
    counts={target:Counter(w for r in records if r['metal']==target for w in r['wavelengths'])
            for target in ['Sn','Bi']}
    shared=set(counts['Sn']) & set(counts['Bi'])
    def curve_points(selected):
        selected=sorted(selected)
        indices=np.array([np.abs(waves-w).argmin() for w in selected],dtype=int)
        if any(abs(waves[i]-w)>1 for i,w in zip(indices,selected)):
            raise ValueError('Selected bands do not match measured wavenumbers')
        return selected,indices
    for target,direction,color in [('Sn',1,SN),('Bi',-1,P['blue_light'])]:
        counter=counts[target];ws=sorted(counter)
        freq.bar(ws,[counter[w]*direction for w in ws],width=5,color=color,lw=0)
        selected,indices=curve_points(set(counter)-shared)
        sizes=np.array([6+counter[w]*4 for w in selected])
        ax.scatter(waves[indices],mean[indices],s=sizes,color=color,edgecolor='white',linewidth=.25,zorder=4)
    selected,indices=curve_points(shared)
    sizes=np.array([6+max(counts['Sn'][w],counts['Bi'][w])*4 for w in selected])
    for color,left in [(SN,True),(P['blue_light'],False)]:
        ax.scatter(waves[indices],mean[indices],s=sizes,color=color,marker=half_marker(left),linewidth=0,zorder=5)
    ax.scatter(waves[indices],mean[indices],s=sizes,facecolor='none',edgecolor='white',linewidth=.25,zorder=6)
    axes_style(ax,None,'Intensity (a.u.)')
    if mean.min()<0 or mean.max()>3000:
        raise ValueError('Mean spectrum exceeds the shared 0–3000 intensity scale')
    ax.set_ylim(0,3000);ax.set_yticks([0,1500,3000])
    ax.tick_params(axis='x',labelbottom=False,length=0)
    ax.legend(handles=[Line2D([],[],color=INK,lw=.8,label='Mean spectrum'),
                       Patch(facecolor=SN,label='Sn-selected bands'),
                       Patch(facecolor=P['blue_light'],label='Bi-selected bands')],
              loc='upper right',bbox_to_anchor=(1,1.70),handlelength=1.2,labelspacing=.25,borderpad=.15)
    freq.axhline(0,color=INK,lw=.5)
    axes_style(freq,WAVE_LABEL,'Frequency')
    freq.set_ylim(-5.5,5.5);freq.set_yticks([-5,0,5]);freq.set_yticklabels(['Bi 5','0','Sn 5'])
    freq.yaxis.labelpad=22
    return freq


def panel_label(ax, letter):
    ax.annotate(f'({letter})', xy=(0,1), xycoords='axes fraction', xytext=(-30,16),
                textcoords='offset points', fontweight='bold', fontsize=12,
                va='bottom', annotation_clip=False)


def build_figure2(raw, processed):
    waves, values = spectra(raw)
    cropped_waves, cropped_values = spectra(raw,400,1800)
    assert len(raw)==len(processed)==187
    fig=plt.figure(figsize=(183/25.4,200/25.4))
    grid=fig.add_gridspec(7,2,left=.10,right=.985,bottom=.14,top=.925,
                         hspace=.70,wspace=.38, height_ratios=[1,1,.25,1,1,1,1])
    a=fig.add_subplot(grid[:2,0])
    b=fig.add_subplot(grid[:2,1])
    for ax,column,target,color,letter in [
        (a,'118Sn (KED)','Sn',P['blue_dark'],'a'),
        (b,'209Bi (KED)','Bi',P['blue_light'],'b')]:
        y=processed[column].to_numpy(dtype=float)
        assert np.isfinite(y).all() and y.min()>=0 and y.max()<=210
        response_panel(ax,y,target,color,'raincloud',letter)
    c=fig.add_subplot(grid[3:,0])
    panel_label(c,'c')
    spectral_overview(c,waves,values)
    c.set_title('Raw spectral variability',loc='left',pad=9)
    c.legend(handles=c.get_legend_handles_labels()[0],
             labels=['All samples','Plus/minus 1 sigma','Mean'],
             loc='upper right',ncol=1,handlelength=1.2)
    stage_axes=[]
    for i,(curve,name,color) in enumerate(zip(
        stages(cropped_values),
        ['Step 1: Raw Data','Step 2: SNV','Step 3: SG (1st Deriv)','Step 4: Standardized'],
        [P['darkest'],P['blue_dark'],P['blue_light'],P['cyan']])):
        ax=fig.add_subplot(grid[i+3,1])
        stage_axes.append(ax)
        ax.plot(cropped_waves,curve,color=color,lw=.8)
        axes_style(ax,WAVE_LABEL if i==3 else None)
        ax.set_xlim(400,1800)
        ax.set_title(name,loc='left',pad=3)
        if i<3:
            ax.tick_params(labelbottom=False)
    upper = stage_axes[0].get_position()
    lower = stage_axes[-1].get_position()
    fig.text(upper.x0 - 42/(fig.get_figwidth()*72),
             (upper.y1 + lower.y0)/2, 'Intensity',
             rotation=90, ha='center', va='center', fontsize=11)
    panel_label(stage_axes[0],'d')
    fig.text(.10,.032,'a–b: KDE (Scott bandwidth), median/IQR, 1.5×IQR whiskers and all 187 observations.',fontsize=9)
    fig.text(.10,.012,'c: mean ± 1 sample SD and full range. d: representative sample Y1; intensity in arbitrary units.',fontsize=9)
    fig._alignment_options = {'exemptions': [{'panels': ['b'],
        'checks': ['vertical-gutter'],
        'reason': 'An intentional spacer separates response distributions from the preprocessing stack; the four stage gutters remain equal.'}]}
    return fig



def main_figures(raw,processed,folds,sweep,performance,selections,global_df,fusion_df):
    waves,vals=spectra(raw);cw,cv=spectra(raw,400,1800)
    # Fig2 uses the selected raincloud distributions and full preprocessing layout.
    fig=build_figure2(raw,processed)
    export(fig,'main/Fig2_data_and_preprocessing','Response distributions, spectral variability and four preprocessing stages.',True)
    # Fig3 retains the feature sweep, five fold-specific means and selection frequencies.
    fig=plt.figure(figsize=(183*MM,260*MM))
    gs=fig.add_gridspec(4,2,left=.10,right=.98,bottom=.065,top=.90,wspace=.36,hspace=.98,height_ratios=[1.15,1,1,1])
    ax=fig.add_subplot(gs[0,:]);plot_sweep(ax,sweep);label(ax,'a')
    merged=raw.merge(folds,left_on=raw.columns[0],right_on='Sample',validate='one_to_one')
    assert len(merged)==len(raw)==187
    for i in range(5):
        ax=fig.add_subplot(gs[1+i//2,i%2]);label(ax,string.ascii_lowercase[i+1]);w,v=spectra(merged[merged.OuterFold!=i+1],400,1800)
        fold_plot(ax,w,v,selections,i+1)
    ax=fig.add_subplot(gs[3,1]);ax.set_gid('frequency');label(ax,'g');frequency_plot(ax,cw,cv,selections)
    fig._alignment_options = {
        'row_groups': [['fold1_spectrum','fold2_spectrum'],['fold3_spectrum','fold4_spectrum'],['fold5_spectrum','frequency_spectrum'],
                       ['fold1_bands','fold2_bands'],['fold3_bands','fold4_bands'],['fold5_bands','frequency_bands']],
        'column_groups': [['fold1_spectrum','fold3_spectrum','fold5_spectrum'],
                          ['fold2_spectrum','fold4_spectrum','frequency_spectrum']]
                          + [[f'fold{i}_spectrum',f'fold{i}_bands'] for i in range(1,6)] + [['frequency_spectrum','frequency_bands']]}
    export(fig,'main/Fig3_band_selection','Feature-count sensitivity and fold selections show compact but partition-dependent Raman support.',True)
    fig,axes=plt.subplots(4,2,figsize=(183*MM,235*MM))
    fig.subplots_adjust(left=.10,right=.98,bottom=.055,top=.91,hspace=1.15,wspace=.4)
    tasks=[('DATA_SN','Sn','train'),('DATA_SN_TEST','Sn','test'),('DATA_BI','Bi','train'),('DATA_BI_TEST','Bi','test')]
    for i,(key,target,subset) in enumerate(tasks):
        ax=axes.flat[i];label(ax,string.ascii_lowercase[i]);plot_boxes(ax,performance[key],target,subset)
    for i,(frame,target,model) in enumerate([(global_df,'Sn','GlobalStack'),(fusion_df,'Sn','MoE'),(global_df,'Bi','GlobalStack'),(fusion_df,'Bi','MoE')]):
        ax=axes.flat[i+4];label(ax,string.ascii_lowercase[i+4]);plot_prediction(ax,frame,target,model)
    legend=[Line2D([],[],ls='',marker=m,color=SN,markersize=3,label=f'Fold {i+1}') for i,m in enumerate(['o','s','^','D','v'])]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.52,.993),ncol=5)
    export(fig,'main/Fig4_model_performance','Fold distributions and OOF predictions bound the performance of the ensemble.',True)
    fig,axes=plt.subplots(2,2,figsize=(183*MM,150*MM))
    fig.subplots_adjust(left=.10,right=.98,bottom=.08,top=.88,hspace=.7,wspace=.35)
    for i,(frame,target,model) in enumerate([(global_df,'Sn','GlobalStack'),(fusion_df,'Sn','MoE'),(global_df,'Bi','GlobalStack'),(fusion_df,'Bi','MoE')]):
        label(axes.flat[i],string.ascii_lowercase[i]);plot_prediction(axes.flat[i],frame,target,model)
    export(fig,'performance/OOF_predictions','The currently supplied OOF table supports paired Sn and Bi predictions.',True)
    fig,ax=new_single(183,78);plot_sweep(ax,sweep)
    fig.subplots_adjust(left=.09,bottom=.18,top=.77)
    export(fig,'spectral_panels/feature_sweep','Shared and single-target feature selection have distinct feature-count sensitivities.',True)
    for key,target,subset in tasks:
        fig,ax=new_single(120,76);fig.subplots_adjust(left=.12,bottom=.32,top=.82)
        plot_boxes(ax,performance[key],target,subset)
        export(fig,f'performance/{target}_{subset}_R2','Median, IQR and all available fold scores compare model configurations.')
    for frame,target,model in [(global_df,'Sn','GlobalStack'),(fusion_df,'Sn','MoE'),(global_df,'Bi','GlobalStack'),(fusion_df,'Bi','MoE')]:
        fig,ax=new_single(89,86);fig.subplots_adjust(left=.19,bottom=.18,top=.78);plot_prediction(ax,frame,target,model)
        export(fig,f'performance/{target}_{model}_OOF','All 187 OOF observations with identity and descriptive fitted lines.')
    fig,ax=new_single(120,75);fig.subplots_adjust(left=.14,top=.78);spectral_overview(ax,waves,vals)
    export(fig,'spectral_panels/raw_spectra_overview','Sample range and sample SD describe all 187 measured spectra.')
    for fold in range(1,6):
        fig,ax=new_single(120,72);fig.subplots_adjust(left=.14,top=.74)
        w,v=spectra(merged[merged.OuterFold!=fold],400,1800);fold_plot(ax,w,v,selections,fold)
        export(fig,f'spectral_panels/selected_bands_fold{fold}','Selected-band lanes preserve both targets and the fold-specific training mean.')
    fig,ax=new_single(120,72);frequency_plot(ax,cw,cv,selections)
    export(fig,'spectral_panels/selection_frequency','Number of outer folds selecting each recorded band; Sn up and Bi down.')
    fig,axs=plt.subplots(4,1,figsize=(89*MM,115*MM),sharex=True)
    fig.subplots_adjust(left=.19,right=.97,bottom=.11,top=.91,hspace=.75)
    for i,(a,data,title,color) in enumerate(zip(axs,stages(cv),['Raw spectrum','SNV','SG first derivative','Standardized'],[INK,SN,P['blue_light'],P['cyan']])):
        a.plot(cw,data,lw=.8,color=color);axes_style(a,WAVE_LABEL if i==3 else None,'a.u.');a.set_title(title,loc='left',pad=3);label(a,string.ascii_lowercase[i])
    export(fig,'spectral_panels/preprocessing','The representative first spectrum follows the original plotting transformations.')
    pre=pd.read_csv(DATA/'processed_all.csv');pw,pv=spectra(pre,400,1800)
    representative=pv[1];scaled=(representative-representative.min())/np.ptp(representative)
    for mode in ['preprocessed','top40','spectral_zones']:
        fig,ax=new_single(89,60)
        if mode=='spectral_zones':add_zones(ax)
        ax.plot(pw,scaled,color=INK,lw=.7)
        if mode=='top40':
            targets=json.loads((DATA/'top40_bands.json').read_text());ids=[np.abs(pw-t).argmin() for t in targets]
            ax.scatter(pw[ids],scaled[ids],s=9,color=SN)
        axes_style(ax,WAVE_LABEL,'Scaled intensity');ax.set_xlim(400,1800)
        export(fig,f'spectral_panels/{mode}','Preserved representative spectrum and the original 40-band illustration; not the final selected support.')
    fig,ax=new_single(120,27);fig.subplots_adjust(left=.02,right=.98,bottom=.05,top=.95);ax.set_xlim(400,1800);ax.set_ylim(0,1);ax.axis('off')
    for lo,hi,name,color in [(400,900,'LFR',P['light']),(900,1400,'MFR',P['green_light']),(1400,1800,'HFR',P['green'])]:
        ax.barh(.5,hi-lo,left=lo,height=.8,color=color,edgecolor='none');ax.text((lo+hi)/2,.62,name,ha='center',va='center');ax.text((lo+hi)/2,.32,f'{lo}–{hi} cm⁻¹',ha='center',va='center')
    export(fig,'spectral_panels/spectral_zone_key','Three Raman fingerprint regions use the original boundaries.')


def appendix(raw,inventory):
    waves,vals=spectra(raw)
    names=raw.iloc[:,0].astype(str).tolist()
    grouped={}
    for i,(name,value) in enumerate(zip(names,vals)):
        sample=int(re.search(r'\d+',name).group())
        polymer=inventory.get(sample,'Unassigned')
        grouped.setdefault(polymer,[]).append((name,value))
        fig,ax=new_single(89,60);ax.plot(waves,value,color=SN,lw=.65)
        axes_style(ax,WAVE_LABEL,'Intensity (a.u.)');ax.set_xlim(waves.min(),waves.max());ax.set_title(f'{name} · {polymer}',loc='left',pad=8)
        export(fig,f'single_spectra/{polymer}/{name}','Original unnormalized measured spectrum; all 2033 numeric spectral columns retained.')
    from supplementary import draw_si
    fig = draw_si(raw, inventory)
    export(fig, 'appendix/FigS1_all_spectra',
           'All 187 spectra in the original 19-polymer-row, 10-sample-column arrangement.')


def write_notes(raw,performance):
    pd.DataFrame(METRICS).drop_duplicates().to_csv(DATA/'recomputed_metrics.csv',index=False)
    missing=[]
    for group,models in performance.items():
        for model,scores in models.items():
            for fold,v in enumerate(scores,1):
                if v is None:missing.append({'group':group,'model':model,'fold':fold})
    (DATA/'missing_regional_scores.json').write_text(json.dumps(missing,indent=2))
    mapping=[]
    for name in raw.iloc[:,0].astype(str):
        polymer=json.loads((DATA/'polymer_inventory.json').read_text()).get(re.search(r'\d+',name).group(),'Unassigned')
        mapping.append({'sample':name,'polymer':polymer,'pdf':f'output/single_spectra/{polymer}/{name}.pdf'})
    pd.DataFrame(mapping).to_csv(DATA/'sample_pdf_map.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-appendix',action='store_true')
    parser.add_argument('--refresh-data', action='store_true', help='Refresh snapshots from the original project data')
    parser.add_argument('--skip-options',action='store_true',help='Skip response-distribution alternatives')
    args=parser.parse_args()
    apply_nature_style()
    performance,selections,inventory=prepare_data(args.refresh_data)
    raw=pd.read_excel(DATA/'raw_spectra.xlsx');processed=pd.read_csv(DATA/'processed_model.csv')
    folds=pd.read_excel(DATA/'folds.xlsx');sweep=pd.read_excel(DATA/'feature_sweep.xlsx')
    global_df=pd.read_excel(DATA/'oof_predictions.xlsx',sheet_name='GlobalStack')
    fusion_df=pd.read_excel(DATA/'oof_predictions.xlsx',sheet_name='MoE')
    assert len(raw)==len(processed)==len(global_df)==len(fusion_df)==187
    main_figures(raw,processed,folds,sweep,performance,selections,global_df,fusion_df)
    if not args.skip_appendix:appendix(raw,inventory)
    write_notes(raw,performance)
    if not args.skip_options:
        from response_options import main as draw_response_options
        draw_response_options()

if __name__=='__main__':main()
