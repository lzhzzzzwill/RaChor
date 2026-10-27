"""Shared preprocessing, Bayesian search, and OOF output for independent baseline entry points."""
import argparse
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
import sys

# Shared RaChor utilities live one directory above the independent baselines.
TRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from config import Config
from data_preprocessing.spectral_preprocessing import apply_snv, SG_WINDOW, SG_POLY, SG_DERIV
from utils.metrics import MetricsUtil

ROOT = Path(__file__).resolve().parent
TARGETS = ["118Sn (KED)", "209Bi (KED)"]


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_spectra(path):
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    for target in TARGETS:
        if target not in frame:
            raise ValueError(f"Missing response column: {target}")
    bands = []
    for col in frame.columns:
        try:
            value = float(col)
        except (TypeError, ValueError):
            continue
        if 400 <= value <= 1800:
            bands.append((value, col))
    bands.sort(key=lambda item: item[0])
    waves = np.asarray([wave for wave, _ in bands])
    if len(waves) < SG_WINDOW or len(np.unique(waves)) != len(waves):
        raise ValueError("Need at least 9 distinct spectral bands in 400–1800 cm^-1")
    labels = frame[TARGETS].to_numpy(dtype=float)
    mask = ~np.any(np.isnan(labels), axis=1)
    used = frame.loc[mask].copy()
    raw = used[[col for _, col in bands]].to_numpy(dtype=float)
    Y = labels[mask]
    if not np.isfinite(raw).all() or not np.isfinite(Y).all():
        raise ValueError("Nonfinite spectral intensities or target values")
    # These operations are per spectrum and do not estimate cross-sample parameters.
    X = savgol_filter(apply_snv(raw), window_length=SG_WINDOW,
                     polyorder=SG_POLY, deriv=SG_DERIV, axis=1)
    names = [col for col in frame.columns if col not in TARGETS and col != "Type"
             and col not in [col for _, col in bands]]
    id_col = next((col for col in ["Sample", "SampleID", "wavelength", "wavenumbers"]
                   if col in used), names[0] if names else None)
    samples = (used[id_col].astype(str).to_numpy() if id_col is not None
               else np.asarray([f"Sample_{i+1}" for i in used.index]))
    info = pd.DataFrame({"Sample": samples, "SourceRow": used.index.to_numpy() + 2})
    if "Type" in used:
        info["Type"] = used["Type"].to_numpy()
    return X, Y, info, waves






def metric_values(y, pred):
    r2, rmse, rpd = MetricsUtil.compute_metrics(y, pred)
    return {"R2": float(r2), "RMSE": float(rmse), "RPD": float(rpd),
            "MAE": float(mean_absolute_error(y, pred))}


def run(args, model_name, bounds, make_pipeline, integer_params=()):
    source, output = local_path(args.data), local_path(args.output_dir)
    X, Y, info, waves = load_spectra(source)
    splits = list(KFold(n_splits=Config.OUTER_SPLITS, shuffle=True,
                        random_state=Config.RANDOM_STATE).split(X, Y))
    if any(len(tr) < 2 * Config.INNER_SPLITS or len(te) < 2 for tr, te in splits):
        raise ValueError("Too few samples for the configured outer/inner R2 evaluation")
    if output.exists() and any(output.iterdir()):
        output = output.with_name(output.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        print(f"[Output] Previous results retained. New run: {output}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running", "targets": TARGETS, "samples": len(info),
        "input_file": args.data, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n_bands": len(waves), "wavenumbers": waves.tolist(), "crop": [400, 1800],
        "snv": True, "sg": {"window": SG_WINDOW, "polyorder": SG_POLY, "deriv": SG_DERIV},
        "standard_scaler": "fitted within each inner/outer training partition",
        "outer_splits": Config.OUTER_SPLITS, "inner_splits": Config.INNER_SPLITS,
        "outer_seed": Config.RANDOM_STATE, "inner_seed": "RANDOM_STATE + outer fold",
        "optimization": "Bayesian optimization, negative mean inner-fold MSE",
        "init_points": args.init_points, "n_iter": args.n_iter, "bounds": bounds, "model": model_name,
        "rpd": "std(y_true, ddof=1) / RMSE",
        "cnn_epochs": getattr(args, "epochs", None),
    }
    status_path = output / "run_metadata.json"
    status_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Input] {len(info)} samples, {len(waves)} bands, "
          f"{waves.min():.2f}–{waves.max():.2f} cm^-1", flush=True)
    predictions = np.full(Y.shape, np.nan)
    fold_ids = np.zeros(len(info), dtype=int)
    metrics, per_fold, tuning, summaries = [], [], [], []
    for fold, (tr, te) in enumerate(splits, 1):
        fold_ids[te] = fold
        for j, target in enumerate(TARGETS):
            short = "Sn" if j == 0 else "Bi"
            print(f"[Outer {fold}/{Config.OUTER_SPLITS}] {target}", flush=True)
            best, trials, score = optimize(X[tr], Y[tr, j], fold, args, bounds, make_pipeline, integer_params)
            tuning.append({"Fold": fold, "Target": target, "BestParams": best,
                           "BestNegativeMSE": score, "Trials": trials})
            (output / "optimization_results.json").write_text(json.dumps(tuning, indent=2), encoding="utf-8")
            pipeline = make_pipeline(best, args.jobs)
            pipeline.fit(X[tr], Y[tr, j])
            for label, idx in [("Train", tr), ("Test", te)]:
                pred = pipeline.predict(X[idx]).reshape(-1)
                if not np.isfinite(pred).all():
                    raise FloatingPointError("Nonfinite baseline predictions")
                metrics.append({"Fold": fold, "Target": target, "Model": model_name + "_BO",
                                "Set": label, **metric_values(Y[idx, j], pred)})
                block = info.iloc[idx].copy()
                block["OuterFold"], block["Target"] = fold, target
                block["Experimental"], block["Predicted"] = Y[idx, j], pred
                block["TrainTest"] = label
                per_fold.append(block)
                if label == "Test":
                    predictions[idx, j] = pred
            if args.save_fold_models:
                with (output / f"{short}_outer_fold_{fold}.pkl").open("wb") as handle:
                    pickle.dump({"pipeline": pipeline, "wavenumbers": waves,
                                 "target": target, "preprocessing": "crop -> apply_snv -> SG then pipeline",
                                 "sg_window": SG_WINDOW, "sg_polyorder": SG_POLY, "sg_deriv": SG_DERIV}, handle)
            pd.DataFrame(metrics).to_csv(output / "fold_performance.csv", index=False)
            pd.concat(per_fold, ignore_index=True).to_csv(output / "per_fold_predictions.csv", index=False)
            progress = info.copy()
            progress["OuterFold"] = fold_ids
            for k, name in enumerate(TARGETS):
                progress[f"{name}_true"] = Y[:, k]
                progress[f"{name}_pred"] = predictions[:, k]
            progress.to_csv(output / "oof_predictions_progress.csv", index=False)
    if not np.isfinite(predictions).all():
        raise ValueError("Some samples have no valid OOF predictions")
    folds = pd.DataFrame(metrics)
    oof = info.copy()
    oof["OuterFold"] = fold_ids
    for j, target in enumerate(TARGETS):
        short = "Sn" if j == 0 else "Bi"
        oof[f"{target}_true"], oof[f"{target}_pred"] = Y[:, j], predictions[:, j]
        single = info.copy()
        single["OuterFold"] = fold_ids
        single["Experimental"], single["Predicted"] = Y[:, j], predictions[:, j]
        single.to_csv(output / f"{short}_oof_predictions.csv", index=False)
        train = folds[(folds.Target == target) & (folds.Set == "Train")]
        row = {"Target": target, "Model": model_name + "_BO"}
        row.update({f"Train {key}": float(train[key].mean()) for key in ["R2", "RMSE", "RPD", "MAE"]})
        row.update({f"OOF {key}": value for key, value in metric_values(Y[:, j], predictions[:, j]).items()})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    summary.to_csv(output / "global_summary.csv", index=False)
    with pd.ExcelWriter(output / "baseline_results.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Global_Summary", index=False)
        oof.to_excel(writer, sheet_name="OOF_Predictions", index=False)
        folds.to_excel(writer, sheet_name="Fold_Performance", index=False)
        pd.concat(per_fold, ignore_index=True).to_excel(writer, sheet_name="Per_Fold_Predictions", index=False)
    plot_dir = output / "plot_data"
    plot_dir.mkdir(exist_ok=True)
    oof.to_csv(plot_dir / "oof_predictions.csv", index=False)
    folds.to_csv(plot_dir / "fold_performance.csv", index=False)
    summary.to_csv(plot_dir / "global_summary.csv", index=False)
    metadata["status"] = "complete"
    status_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved to {output}", flush=True)






def optimize(X, y, fold, args, bounds, make_pipeline, integer_params):
    inner = list(KFold(n_splits=Config.INNER_SPLITS, shuffle=True,
                       random_state=Config.RANDOM_STATE + fold).split(X))

    def objective(**candidate):
        params = dict(candidate)
        for key in integer_params:
            params[key] = int(params[key])
        losses = []
        for tr, va in inner:
            pipeline = make_pipeline(params, args.jobs)
            pipeline.fit(X[tr], y[tr])
            prediction = pipeline.predict(X[va]).reshape(-1)
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Nonfinite inner validation predictions")
            losses.append(mean_squared_error(y[va], prediction))
        return -float(np.mean(losses))

    optimizer = BayesianOptimization(f=objective, pbounds=bounds,
                                    random_state=Config.RANDOM_STATE, verbose=2)
    optimizer.maximize(init_points=args.init_points, n_iter=args.n_iter)
    best = dict(optimizer.max["params"])
    for key in integer_params:
        best[key] = int(best[key])
    return best, optimizer.res, float(optimizer.max["target"])


def parse_args(model_name):
    parser = argparse.ArgumentParser(description=f"Independent Bayesian {model_name}: Sn/Bi spectral OOF evaluation")
    parser.add_argument("--data", default="../data/Raman_spectroscopy_data.xlsx", help="Raw .xlsx or .csv")
    parser.add_argument("--output-dir", default=f"results/{model_name.lower()}")
    parser.add_argument("--init-points", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--save-fold-models", action="store_true")
    if model_name == "CNN":
        parser.add_argument("--epochs", type=int, default=150,
                            help="Training epochs per inner/outer fit; never selected using outer test labels")
    args = parser.parse_args()
    if args.init_points < 1 or args.n_iter < 0 or args.jobs == 0 or getattr(args, "epochs", 1) < 1:
        parser.error("Use positive init-points/epochs, nonnegative n-iter, and nonzero jobs")
    return args
