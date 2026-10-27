"""Figure-3-style fold distributions and pooled OOF comparisons from real runs.

No training, simulation, filtering by score, or downsampling is performed here.
All CLI paths are relative to the RaChor directory unless explicitly absolute.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
STYLE_DIR = BASE / "plotting_helpers"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))
cache = BASE / ".cache" / "baseline_comparison"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from publication_style import PALETTE as P, apply_nature_style
from audit_panel_alignment import require_matplotlib_panel_alignment
from audit_figure_collisions import audit_pdf
from audit_pdf_text import audit_pdf as audit_text

TARGETS = {"Sn": "118Sn (KED)", "Bi": "209Bi (KED)"}
ORDER = ["RF", "XGBoost", "CatBoost", "CNN"]
COLORS = [P["blue_dark"], P["blue"], P["cyan"], P["darkest"], P["blue_light"]]
MARKERS = ["o", "s", "^", "D", "v"]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def completed_run(prefix):
    """Use latest COMPLETE run by modification time, never by best score."""
    candidates = [prefix] + list(prefix.parent.glob(prefix.name + "_*"))
    valid = []
    for path in candidates:
        meta = path / "run_metadata.json"
        if not meta.is_file():
            continue
        data = json.loads(meta.read_text())
        if data.get("status") == "complete":
            valid.append((meta.stat().st_mtime_ns, path))
    if not valid:
        raise FileNotFoundError(f"No completed run under {prefix}; train this model or provide its --*-dir")
    return max(valid, key=lambda pair: pair[0])[1]


def canonical_oof(frame, model):
    required = ["Sample", "OuterFold"] + [f"{t}_{suffix}" for t in TARGETS.values() for suffix in ["true", "pred"]]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{model}: missing OOF columns {sorted(missing)}")
    frame = frame.copy()
    frame["Sample"] = frame.Sample.astype(str)
    if frame.Sample.duplicated().any():
        raise ValueError(f"{model}: duplicate Sample identifiers; fix matching before plotting")
    for col in required[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(float)).all():
            raise ValueError(f"{model}: nonfinite values in {col}")
    if not np.array_equal(frame.OuterFold, frame.OuterFold.astype(int)):
        raise ValueError(f"{model}: noninteger fold identifiers")
    frame["OuterFold"] = frame.OuterFold.astype(int)
    if sorted(frame.OuterFold.unique()) != [1, 2, 3, 4, 5]:
        raise ValueError(f"{model}: expected five complete outer folds")
    return frame


def load_sources(args):
    paths = {"RF": args.rf_dir, "XGBoost": args.xgb_dir,
             "CatBoost": args.catboost_dir, "CNN": args.cnn_dir}
    frames, folds, provenance = {}, [], []
    for model in args.models:
        supplied = paths[model]
        if supplied is not None:
            path = project_path(supplied)
        else:
            prefix = (BASE / "results/xgb_bayesian_oof"
                      if model == "XGBoost" else BASE / f"results/{model.lower()}")
            path = completed_run(prefix)
        meta = json.loads((path / "run_metadata.json").read_text())
        if meta.get("status") != "complete":
            raise ValueError(f"Incomplete run: {path}")
        expected = model if model != "XGBoost" else None
        if expected and meta.get("model") != expected:
            raise ValueError(f"Model metadata mismatch: {path}")
        frames[model] = canonical_oof(pd.read_csv(path / "oof_predictions.csv"), model)
        performance = pd.read_csv(path / "fold_performance.csv")
        if set(performance.Model) != {model + "_BO"}:
            raise ValueError(f"Unexpected model labels in {path}")
        performance["Model"] = model
        folds.append(performance)
        provenance.append({"model": model, "directory": os.path.relpath(path, PROJECT),
                           "oof_sha256": hashlib.sha256((path / "oof_predictions.csv").read_bytes()).hexdigest(),
                           "training_metadata": meta})
    if not args.no_rachor:
        path = project_path(args.rachor_results)
        with pd.ExcelFile(path) as workbook:
            candidates = ["MoE_Predictions", "MoE"]
            sheet = next((name for name in candidates if name in workbook.sheet_names), None)
            if sheet is None:
                raise ValueError("RaChor workbook has no final-fusion OOF predictions")
            frames["RaChor"] = canonical_oof(pd.read_excel(workbook, sheet_name=sheet), "RaChor")
            perf = pd.read_excel(workbook, sheet_name="Fold_Performance")
            perf = perf[perf.Model == "MoE"].copy()
            perf["Model"] = "RaChor"
            folds.append(perf)
        provenance.append({"model": "RaChor", "file": os.path.relpath(path, PROJECT),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "preprocessing_note": "existing RaChor results; original full-dataset spectral scaling"})
    anchor = next(iter(frames.values())).set_index("Sample")
    for model, frame in frames.items():
        indexed = frame.set_index("Sample")
        if set(indexed.index) != set(anchor.index):
            raise ValueError(f"{model}: sample set differs; unmatched data cannot be compared")
        indexed = indexed.loc[anchor.index]
        if not np.array_equal(anchor.OuterFold, indexed.OuterFold):
            raise ValueError(f"{model}: outer-fold assignments differ")
        for target in TARGETS.values():
            if not np.allclose(anchor[target + "_true"], indexed[target + "_true"], rtol=0, atol=1e-10):
                raise ValueError(f"{model}: measured labels differ for {target}")
        frames[model] = indexed.reset_index()
    fold_frame = pd.concat(folds, ignore_index=True)
    metrics, long = [], []
    for model, frame in frames.items():
        for short, target in TARGETS.items():
            true = frame[target + "_true"].to_numpy(float)
            pred = frame[target + "_pred"].to_numpy(float)
            rmse = np.sqrt(np.mean((true - pred) ** 2))
            sst = np.sum((true - true.mean()) ** 2)
            if sst <= 0:
                raise ValueError(f"Constant measured target: {target}")
            metrics.append({"Model": model, "Target": target, "n": len(true),
                            "OOF R2": 1 - np.sum((true - pred) ** 2) / sst,
                            "OOF RMSE": rmse,
                            "OOF RPD": true.std(ddof=1) / rmse if rmse else np.inf,
                            "OOF MAE": np.mean(np.abs(true - pred))})
            long.append(pd.DataFrame({"Sample": frame.Sample, "OuterFold": frame.OuterFold,
                                      "Model": model, "Target": target,
                                      "Measured": true, "Predicted": pred}))
            for label in ["Train", "Test"]:
                group = fold_frame[(fold_frame.Model == model) & (fold_frame.Target == target)
                                   & (fold_frame.Set == label)]
                if len(group) != 5 or sorted(group.Fold) != [1, 2, 3, 4, 5]:
                    raise ValueError(f"{model}/{target}/{label}: need five distinct fold scores")
                if not np.isfinite(group[["R2", "RMSE"]].to_numpy(float)).all():
                    raise ValueError(f"{model}: invalid fold metrics")
                if label == "Test":
                    for row in group.itertuples():
                        mask = frame.OuterFold.to_numpy() == row.Fold
                        actual_rmse = np.sqrt(np.mean((true[mask] - pred[mask]) ** 2))
                        actual_r2 = 1 - np.sum((true[mask] - pred[mask]) ** 2) / np.sum((true[mask] - true[mask].mean()) ** 2)
                        if not np.allclose([row.RMSE, row.R2], [actual_rmse, actual_r2], rtol=1e-6, atol=1e-8):
                            raise ValueError(f"{model}: fold table and OOF predictions disagree")
    return pd.concat(long, ignore_index=True), fold_frame, pd.DataFrame(metrics), provenance


def axes_style(ax, letter):
    ax.text(-.16, 1.08, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax.tick_params(direction="out", length=3, width=.8, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def save_checked(fig, prefix, qa):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(qa / (prefix.name + ".alignment.json")),
                                      tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".svg"))
    fig.savefig(prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)
    collision = audit_pdf(prefix.with_suffix(".pdf"))
    (qa / (prefix.name + ".collision.json")).write_text(json.dumps(collision, indent=2), encoding="utf-8")
    text = audit_text(prefix.with_suffix(".pdf").read_bytes())
    (qa / (prefix.name + ".text.json")).write_text(json.dumps(text, indent=2), encoding="utf-8")
    if collision["summary"]["fail"] or text.get("below_minimum_count", 0) or not text.get("auditable", False):
        raise RuntimeError(f"Figure requires layout/text repair; inspect {qa}")
    if collision["summary"]["warn"]:
        print(f"[QA] Review collision warnings: {prefix.name}", flush=True)


def draw_folds(folds, order, output, qa):
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 170 / 25.4))
    fig.subplots_adjust(left=.11, right=.98, bottom=.12, top=.89, wspace=.38, hspace=.68)
    for row, (short, target) in enumerate(TARGETS.items()):
        for col, label in enumerate(["Train", "Test"]):
            ax = axes[row, col]
            groups = [folds[(folds.Model == model) & (folds.Target == target) & (folds.Set == label)]
                      .sort_values("Fold").R2.to_numpy(float) for model in order]
            boxes = ax.boxplot(groups, positions=np.arange(len(order)) + 1, widths=.58,
                               patch_artist=True, showfliers=False, whis=1.5,
                               medianprops={"color": P["darkest"], "linewidth": 1},
                               boxprops={"linewidth": .6, "edgecolor": P["darkest"]},
                               whiskerprops={"linewidth": .6}, capprops={"linewidth": .6})
            for i, (model, values) in enumerate(zip(order, groups)):
                boxes["boxes"][i].set_facecolor(P["green"] if model == "RaChor" else P["light"])
                for k, score in enumerate(values):
                    ax.scatter(i + 1 + (k - 2) * .075, score, s=16, marker=MARKERS[k],
                               facecolor=P["blue_dark"], edgecolor="white", linewidth=.25, zorder=3)
            ax.set_xticks(np.arange(len(order)) + 1, order, rotation=35, ha="right", rotation_mode="anchor")
            values = np.concatenate(groups)
            low = min(.6, float(values.min()) - .04)
            ax.set_ylim(low, max(1.01, float(values.max()) + .025))
            ax.set_ylabel("$R^2$")
            ax.set_title(f"{short} · {'Training' if label == 'Train' else 'Held-out folds'}", loc="left", pad=10)
            axes_style(ax, chr(97 + row * 2 + col))
            # Restore categorical ticks after common styling.
            ax.set_xticks(np.arange(len(order)) + 1, order, rotation=35, ha="right", rotation_mode="anchor")
    handles = [Line2D([], [], linestyle="", marker=marker, color=P["blue_dark"],
                      markersize=4, label=f"Fold {i + 1}") for i, marker in enumerate(MARKERS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .99), ncol=5, frameon=False)
    save_checked(fig, output / "baseline_fold_performance", qa)


def draw_oof(long, metrics, order, output, qa):
    fig, axes = plt.subplots(2, len(order), squeeze=False,
                             figsize=(max(183, 62 * len(order)) / 25.4, 152 / 25.4))
    fig.subplots_adjust(left=.065, right=.985, bottom=.12, top=.91, wspace=.5, hspace=.65)
    for row, (short, target) in enumerate(TARGETS.items()):
        target_data = long[long.Target == target]
        extent = target_data[["Measured", "Predicted"]].to_numpy()
        low, high = float(extent.min()), float(extent.max())
        pad = max((high - low) * .08, 1e-6)
        limits = np.array([low - pad, high + pad])
        for col, model in enumerate(order):
            ax = axes[row, col]
            data = target_data[target_data.Model == model]
            true, pred = data.Measured.to_numpy(), data.Predicted.to_numpy()
            ax.plot(limits, limits, "--", color="#777777", lw=.7, label="Y = T")
            slope, intercept = np.polyfit(true, pred, 1)
            ax.plot(limits, slope * limits + intercept, color=P["darkest"], lw=.8, label="Fit")
            ax.scatter(true, pred, s=10, color=COLORS[col], edgecolor=P["darkest"], linewidth=.25, alpha=.78)
            score = metrics[(metrics.Model == model) & (metrics.Target == target)].iloc[0]
            annotation = (f"$R^2$: {score['OOF R2']:.4f}\n"
                          f"RMSE: {score['OOF RMSE']:.3f}\nRPD: {score['OOF RPD']:.3f}")
            ax.text(.04, .96, annotation, transform=ax.transAxes, fontsize=9, va="top", linespacing=1.3,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
            ax.set_xlim(limits); ax.set_ylim(limits)
            ax.set_xlabel("Measured response"); ax.set_ylabel("Predicted response")
            ax.set_title(f"{short} · {model}", loc="left", pad=10)
            axes_style(ax, chr(97 + row * len(order) + col))
            ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.5)
    save_checked(fig, output / "baseline_oof_predictions", qa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=ORDER, default=ORDER)
    for name in ["rf", "xgb", "catboost", "cnn"]:
        parser.add_argument(f"--{name}-dir", default=None)
    parser.add_argument("--no-rachor", action="store_true")
    parser.add_argument("--rachor-results", default="nestedcv_oof_predictions.xlsx")
    parser.add_argument("--output-dir", default="other_models/figures")
    parser.add_argument("--data-dir", default="other_models/plot_data")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Model names must be distinct")
    long, folds, metrics, provenance = load_sources(args)
    output, data_dir = project_path(args.output_dir), project_path(args.data_dir)
    qa = output / "qa"
    for path in [output, data_dir, qa]:
        path.mkdir(parents=True, exist_ok=True)
    long.to_csv(data_dir / "oof_plot_data.csv", index=False)
    folds.to_csv(data_dir / "fold_plot_data.csv", index=False)
    metrics.to_csv(data_dir / "pooled_oof_metrics.csv", index=False)
    (data_dir / "source_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (data_dir / "figure_contract.json").write_text(json.dumps({
        "question": "How do independent full-spectrum baselines compare in held-out accuracy and calibration?",
        "style_reference": "Figure 3 model_performance; palette, typography, fold symbols, scatter conventions",
        "reuse_level": "style-only inheritance; metrics use sample SD (ddof=1)",
        "panels": {"folds": "train/test gap and partition variability", "oof": "pooled predictions and calibration"},
        "observations": len(long), "excluded_observations": 0,
        "latest_run_rule": "most recently completed run by metadata modification time; never best performance",
        "statistical_note": "five folds share overlapping training data; no independent-replicate significance test",
    }, indent=2), encoding="utf-8")
    apply_nature_style()
    # Explicit export contract supplements the shared Figure-3 style helper.
    matplotlib.rcParams.update({"font.family": "sans-serif",
                                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                                "svg.fonttype": "none", "pdf.fonttype": 42})
    order = args.models + ([] if args.no_rachor else ["RaChor"])
    draw_folds(folds, order, output, qa)
    draw_oof(long, metrics, order, output, qa)
    print(metrics.to_string(index=False), flush=True)
    print(f"Figures: {output}\nSource data: {data_dir}", flush=True)


if __name__ == "__main__":
    main()
