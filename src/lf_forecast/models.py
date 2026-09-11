"""Recurrent forecasters: encode a window of past hours, decode a direct
multi-horizon forecast in one shot (no autoregressive rollout, so
prediction error does not compound across the horizon).

Architecture note - residual-over-seasonal-naive:
The daily load-factor cycle is strong and regular enough that
seasonal-naive ("tomorrow looks like today") is a hard baseline to beat by
predicting raw values from scratch (verified empirically: an
unconstrained sigmoid-output RNN of this size landed *behind*
seasonal-naive on the test set). Instead the RNN predicts a bounded
*correction* on top of the seasonal-naive reference (the load_factor
values from exactly `horizon` steps back, i.e. the previous day's same
hours, which are already present in its own input window). This lets the
network focus its capacity on the part seasonal-naive gets wrong -
day-to-day deviations - rather than re-deriving the daily shape it can
already read off its own input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LOAD_FACTOR_FEATURE_IDX = 0


class _RNNForecaster(nn.Module):
    def __init__(
        self,
        rnn_cls,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
        correction_scale: float = 0.3,
    ):
        super().__init__()
        self.horizon = horizon
        self.correction_scale = correction_scale
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        correction = self.head(last_hidden) * self.correction_scale

        seasonal_naive_ref = x[:, -self.horizon :, LOAD_FACTOR_FEATURE_IDX]
        pred = seasonal_naive_ref + correction
        return torch.clamp(pred, 0.0, 1.0)


class LSTMForecaster(_RNNForecaster):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float):
        super().__init__(nn.LSTM, input_size, hidden_size, num_layers, horizon, dropout)


class GRUForecaster(_RNNForecaster):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float):
        super().__init__(nn.GRU, input_size, hidden_size, num_layers, horizon, dropout)


def build_model(model_type: str, input_size: int, hidden_size: int, num_layers: int, horizon: int, dropout: float) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "lstm":
        return LSTMForecaster(input_size, hidden_size, num_layers, horizon, dropout)
    if model_type == "gru":
        return GRUForecaster(input_size, hidden_size, num_layers, horizon, dropout)
    raise ValueError(f"Unknown model_type: {model_type!r} (expected 'lstm' or 'gru')")
