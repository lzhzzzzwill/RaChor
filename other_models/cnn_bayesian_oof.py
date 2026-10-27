"""Independent 1-D CNN baseline: Bayesian 20+40 search and nested OOF."""
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from cnn_baseline import CNNRegressor
from baseline_workflow import Config, parse_args, run

# Log coordinates provide a balanced search across orders of magnitude.
BOUNDS = {"channels": (8, 32), "log10_learning_rate": (-4, -2),
          "log10_weight_decay": (-6, -2), "batch_size": (8, 32)}
INTEGER_PARAMS = ["channels", "batch_size"]


def make_pipeline(params, jobs, epochs=150):
    return Pipeline([("scaler", StandardScaler()),
                     ("regressor", CNNRegressor(
                         channels=int(params["channels"]), batch_size=int(params["batch_size"]),
                         learning_rate=10 ** params["log10_learning_rate"],
                         weight_decay=10 ** params["log10_weight_decay"],
                         epochs=epochs, random_state=Config.RANDOM_STATE))])


if __name__ == "__main__":
    args = parse_args("CNN")
    run(args, "CNN", BOUNDS, partial(make_pipeline, epochs=args.epochs), INTEGER_PARAMS)
