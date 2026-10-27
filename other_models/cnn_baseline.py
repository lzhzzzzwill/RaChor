"""A small sklearn-compatible 1-D CNN regressor for spectral comparisons."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CNNRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, channels=16, learning_rate=0.001, weight_decay=0.001,
                 epochs=150, batch_size=32, random_state=42):
        self.channels = channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        from torch import nn

        X, y = check_X_y(X, y, dtype=np.float32)
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        self.n_features_in_ = X.shape[1]
        self.y_mean_ = float(y.mean())
        self.y_scale_ = max(float(y.std()), 1e-8)
        # Small datasets run efficiently on CPU; avoid nested thread pools.
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            pool_size = min(8, self.n_features_in_)
            self.network_ = nn.Sequential(
                nn.Conv1d(1, self.channels, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(self.channels, self.channels * 2, kernel_size=3, padding=1),
                nn.ReLU(), nn.AdaptiveAvgPool1d(pool_size), nn.Flatten(),
                nn.Linear(self.channels * 2 * pool_size, 32), nn.ReLU(),
                nn.Linear(32, 1),
            )
            features = torch.from_numpy(np.ascontiguousarray(X)).unsqueeze(1)
            labels = torch.from_numpy(((y - self.y_mean_) / self.y_scale_).astype(np.float32))
            optimizer = torch.optim.Adam(
                self.network_.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            rng = np.random.default_rng(self.random_state)
            self.network_.train()
            for _ in range(self.epochs):
                order = rng.permutation(len(X))
                for start in range(0, len(X), self.batch_size):
                    idx = order[start:start + self.batch_size]
                    optimizer.zero_grad()
                    prediction = self.network_(features[idx]).squeeze(-1)
                    loss = nn.functional.mse_loss(prediction, labels[idx])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("CNN training loss is not finite")
                    loss.backward()
                    optimizer.step()
        return self

    def predict(self, X):
        import torch

        check_is_fitted(self, "network_")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count differs from fitted CNN")
        self.network_.eval()
        with torch.no_grad():
            values = self.network_(torch.from_numpy(np.ascontiguousarray(X)).unsqueeze(1))
        return values.numpy().reshape(-1) * self.y_scale_ + self.y_mean_
