"""Independent RF baseline: full spectra, Bayesian 20+40 search and nested OOF."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from baseline_workflow import Config, parse_args, run

BOUNDS = {"n_estimators": (100, 1000), "max_depth": (3, 20),
          "min_samples_split": (2, 10), "min_samples_leaf": (1, 8),
          "max_features": (0.3, 1.0)}
INTEGER_PARAMS = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]


def make_pipeline(params, jobs):
    params = dict(params)
    for key in INTEGER_PARAMS:
        params[key] = int(params[key])
    return Pipeline([("scaler", StandardScaler()),
                     ("regressor", RandomForestRegressor(random_state=Config.RANDOM_STATE,
                                                         n_jobs=jobs, **params))])


if __name__ == "__main__":
    run(parse_args("RF"), "RF", BOUNDS, make_pipeline, INTEGER_PARAMS)
