"""Non-deep-learning baselines, evaluated on exactly the same
(window -> horizon) test windows as the LSTM/GRU models for a fair
comparison.

- Seasonal-naive: forecast hour h of the target day = actual value at the
  same hour, `horizon` steps earlier. Since horizon == 24h by default, this
  is simply "tomorrow looks like today" and needs no fitting: for every
  window the last `horizon` steps of the input window (load_factor feature)
  are the prediction.

- Holt-Winters exponential smoothing (additive seasonal, period=24h):
  fit independently on each window's own input history and forecast the
  next `horizon` steps. This is the classical statistical baseline
  requested alongside seasonal-naive.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .features import Windows

LOAD_FACTOR_FEATURE_IDX = 0


def seasonal_naive_predict(windows: Windows, horizon: int) -> np.ndarray:
    return windows.X[:, -horizon:, LOAD_FACTOR_FEATURE_IDX]


def holt_winters_predict(windows: Windows, horizon: int, seasonal_periods: int = 24) -> np.ndarray:
    n = len(windows.X)
    preds = np.zeros((n, horizon), dtype=np.float32)
    for i in range(n):
        history = windows.X[i, :, LOAD_FACTOR_FEATURE_IDX].astype(float)
        try:
            model = ExponentialSmoothing(
                history,
                trend=None,
                seasonal="add",
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
            preds[i] = model.forecast(horizon)
        except Exception:
            # fall back to seasonal-naive for the rare ill-conditioned fit
            preds[i] = history[-horizon:]
    return preds
