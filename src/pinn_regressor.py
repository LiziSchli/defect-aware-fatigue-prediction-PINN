from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass
class TrainingConfig:
    input_dim: int
    hidden_dims: tuple[int, ...]
    learning_rate: float
    weight_decay: float


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PhysicsGuidedRegressor(BaseEstimator, RegressorMixin):
    """A small PINN-style regressor with monotonic physics penalties.

    The penalty encourages predicted fatigue life to decrease as defect
    severity and local stress increase.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 32),
        learning_rate: float = 1e-3,
        epochs: int = 240,
        batch_size: int = 32,
        lambda_physics: float = 0.15,
        weight_decay: float = 1e-4,
        constrained_indices: tuple[int, ...] = (0, 5, 10),
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_physics = lambda_physics
        self.weight_decay = weight_decay
        self.constrained_indices = constrained_indices
        self.random_state = random_state
        self.verbose = verbose

    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _build_model(self, input_dim: int) -> FeedForwardNet:
        config = TrainingConfig(
            input_dim=input_dim,
            hidden_dims=tuple(self.hidden_dims),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.config_ = config
        return FeedForwardNet(input_dim=config.input_dim, hidden_dims=config.hidden_dims)

    def fit(self, x, y):
        self._set_seed()

        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        self.n_features_in_ = x_np.shape[1]

        self.model_ = self._build_model(self.n_features_in_)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        dataset_x = torch.tensor(x_np, dtype=torch.float32)
        dataset_y = torch.tensor(y_np, dtype=torch.float32)

        self.loss_history_ = []
        generator = torch.Generator().manual_seed(self.random_state)

        for epoch in range(self.epochs):
            permutation = torch.randperm(len(dataset_x), generator=generator)
            epoch_loss = 0.0

            for start in range(0, len(dataset_x), self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                xb = dataset_x[batch_idx].clone().detach().requires_grad_(True)
                yb = dataset_y[batch_idx]

                optimizer.zero_grad()
                pred = self.model_(xb)
                mse_loss = criterion(pred, yb)

                grads = torch.autograd.grad(
                    outputs=pred.sum(),
                    inputs=xb,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                penalties = []
                for idx in self.constrained_indices:
                    if idx < grads.shape[1]:
                        penalties.append(torch.relu(grads[:, idx]).mean())

                physics_loss = torch.stack(penalties).mean() if penalties else torch.tensor(0.0)
                loss = mse_loss + self.lambda_physics * physics_loss
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach()) * len(batch_idx)

            mean_epoch_loss = epoch_loss / len(dataset_x)
            self.loss_history_.append(mean_epoch_loss)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss={mean_epoch_loss:.4f}")

        return self

    def predict(self, x):
        x_np = np.asarray(x, dtype=np.float32)
        xt = torch.tensor(x_np, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(xt).cpu().numpy()
        return pred.reshape(-1)
