import argparse
import os
import subprocess
import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
cache_dir = Path(os.environ.get("XDG_CACHE_HOME", ".cache/tabpfn"))
if not cache_dir.is_absolute():
    cache_dir = PROJECT_DIR / cache_dir
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(cache_dir)
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(PROJECT_DIR / ".cache" / "tabpfn"))
from sklearn.exceptions import ConvergenceWarning
from config import Config
from pipeline.orchestrator import NestedCVOrchestrator

os.environ["PYTHONWARNINGS"] = "ignore"

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shared-max-features",
        type=int,
        default=Config.SHARED_MAX_FEATURES,
        help="Override Config.SHARED_MAX_FEATURES for the current run."
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=PROJECT_DIR / "artifacts",
        help="Destination for the full-data MoE model. Relative paths are resolved against the training script directory."
    )
    parser.add_argument(
        "--skip-model-export", action="store_true",
        help="Run nested cross-validation only, without fitting/exporting a full-data model."
    )
    args = parser.parse_args(argv)
    if args.shared_max_features <= 0:
        parser.error("--shared-max-features must be a positive integer.")
    return args

def suppress_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="^Objective did not converge.*")
    warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size .*different to the input size.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
    warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")

def main(argv=None):
    args = parse_args(argv)
    suppress_warnings()
    Config.SHARED_MAX_FEATURES = args.shared_max_features
    
    CURRENT_DIR = PROJECT_DIR
    
    DATA_PATH = CURRENT_DIR / "data" / "Raman_spectroscopy_data_preprocessed.csv"
    OUTPUT_XLSX_PATH = CURRENT_DIR / "nestedcv_oof_predictions.xlsx"
    TARGET_METALS = ["118Sn (KED)", "209Bi (KED)"]

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Preprocessed training CSV not found: {DATA_PATH}. Run preprocessing first.")

    
    print(f"[*] Data Source: {DATA_PATH}")
    print(f"[*] Output Target: {OUTPUT_XLSX_PATH}")
    print(f"[*] SHARED_MAX_FEATURES: {Config.SHARED_MAX_FEATURES}")
    print(f"[*] TARGET_SHARED_MAX_FEATURES: {getattr(Config, 'TARGET_SHARED_MAX_FEATURES', {})}")

    runner = NestedCVOrchestrator(
        data_path=DATA_PATH,
        output_path=OUTPUT_XLSX_PATH,
        target_metals=TARGET_METALS
    )
    runner.run()
    if not args.skip_model_export:
        artifact_dir = args.artifact_dir
        if not artifact_dir.is_absolute():
            artifact_dir = PROJECT_DIR / artifact_dir
        print("[*] Cross-validation finished. Fitting/exporting a full-data MoE model next.", flush=True)
        subprocess.run([
            sys.executable,
            str(PROJECT_DIR / "export_deployment_artifacts.py"),
            "--shared-max-features", str(args.shared_max_features),
            "--artifact-dir", str(artifact_dir.resolve()),
            "--target-metals", *TARGET_METALS,
        ], check=True)


if __name__ == "__main__":
    main()
