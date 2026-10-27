"""Independent CatBoost baseline: Bayesian 20+40 search and nested OOF."""
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from baseline_workflow import Config, parse_args, run

BOUNDS = {"iterations": (100, 1000), "depth": (3, 8),
          "learning_rate": (0.01, 0.3), "l2_leaf_reg": (1, 15),
          "random_strength": (0, 5)}
INTEGER_PARAMS = ["iterations", "depth"]


def make_pipeline(params, jobs):
    params = dict(params)
    for key in INTEGER_PARAMS:
        params[key] = int(params[key])
    return Pipeline([("scaler", StandardScaler()),
                     ("regressor", CatBoostRegressor(
                         random_seed=Config.RANDOM_STATE, thread_count=jobs,
                         loss_function="RMSE", verbose=False,
                         allow_writing_files=False, **params))])


if __name__ == "__main__":
    run(parse_args("CatBoost"), "CatBoost", BOUNDS, make_pipeline, INTEGER_PARAMS)
