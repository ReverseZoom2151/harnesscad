"""Surrogate-model preprocessing, error metrics, and ensemble averaging.

Deterministic pieces of the transfer-learning surrogate stage (Section 5.2) of:

    Yoo et al., "Integrating deep learning into CAD/CAE system: generative design
    and evaluation of 3D conceptual wheel", Struct. Multidisc. Optim. 64 (2021)
    2725-2747.

The learned CNN/transfer-learning surrogate itself is external.  Around it,
though, the paper uses a handful of standard, fully deterministic operations to
prepare labels and to evaluate/aggregate predictions.  This module implements
those:

    * Min-max scaling / normalization (equation 5)::

          y_scale = (y - y_min) / (y_max - y_min)

      used to rescale the frequency and mass output labels to a fixed [0, 1]
      range, plus the inverse transform back to physical units.

    * Root-mean-square error (equation 6)::

          RMSE = sqrt( (1/n) * sum_i (yhat_i - y_i)^2 )

    * Mean absolute percent error (equation 7)::

          MAPE = (100/n) * sum_i | (y_i - yhat_i) / y_i |

    * Ensemble averaging (Section 5.2.2, Fig. 19) -- the final model averages the
      predictions of several frequency models and several mass models.  The
      ensemble prediction is the arithmetic mean of the member predictions.

All functions are deterministic and stdlib-only (``math``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Equation (5): min-max scaling / normalization.
# ---------------------------------------------------------------------------
@dataclass
class MinMaxScaler:
    """A fitted min-max scaler mapping ``[y_min, y_max]`` onto ``[0, 1]``."""

    y_min: float
    y_max: float

    def _span(self) -> float:
        span = self.y_max - self.y_min
        if span == 0.0:
            raise ValueError("cannot scale a constant range (y_max == y_min)")
        return span

    def scale(self, y: float) -> float:
        """Forward transform ``(y - y_min) / (y_max - y_min)`` (equation 5)."""
        return (y - self.y_min) / self._span()

    def inverse(self, y_scaled: float) -> float:
        """Inverse transform back to physical units."""
        return y_scaled * (self.y_max - self.y_min) + self.y_min

    def scale_all(self, values: Sequence[float]) -> List[float]:
        span = self._span()
        return [(v - self.y_min) / span for v in values]


def fit_minmax(values: Sequence[float]) -> MinMaxScaler:
    """Fit a :class:`MinMaxScaler` to ``values``.

    Raises ``ValueError`` for an empty sequence or a constant range.
    """
    if not values:
        raise ValueError("cannot fit min-max scaler to empty data")
    lo = min(values)
    hi = max(values)
    if lo == hi:
        raise ValueError("cannot scale constant data (all values equal)")
    return MinMaxScaler(y_min=float(lo), y_max=float(hi))


# ---------------------------------------------------------------------------
# Equations (6) and (7): surrogate error metrics.
# ---------------------------------------------------------------------------
def _check_pair(predicted: Sequence[float], actual: Sequence[float]) -> int:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have equal length")
    n = len(predicted)
    if n == 0:
        raise ValueError("need at least one data point")
    return n


def rmse(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Root-mean-square error (equation 6)."""
    n = _check_pair(predicted, actual)
    total = 0.0
    for yhat, y in zip(predicted, actual):
        d = yhat - y
        total += d * d
    return math.sqrt(total / n)


def mape(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean absolute percent error, in percent (equation 7).

    Raises ``ValueError`` if any ground-truth value is zero (undefined
    percentage error).
    """
    n = _check_pair(predicted, actual)
    total = 0.0
    for yhat, y in zip(predicted, actual):
        if y == 0.0:
            raise ValueError("MAPE undefined when a ground-truth value is zero")
        total += abs((y - yhat) / y)
    return 100.0 * total / n


# ---------------------------------------------------------------------------
# Ensemble averaging (Section 5.2.2).
# ---------------------------------------------------------------------------
def ensemble_mean(member_predictions: Sequence[float]) -> float:
    """Arithmetic mean of member model predictions for one sample."""
    if not member_predictions:
        raise ValueError("need at least one member prediction")
    return math.fsum(member_predictions) / len(member_predictions)


def ensemble_predict(member_columns: Sequence[Sequence[float]]) -> List[float]:
    """Average an ensemble of per-member prediction vectors.

    ``member_columns`` is a sequence of members, each a sequence of predictions
    over the same ``n`` samples (member-major).  Returns the length-``n`` vector
    of per-sample ensemble means.  Raises ``ValueError`` if members disagree on
    length or the ensemble is empty.
    """
    if not member_columns:
        raise ValueError("need at least one ensemble member")
    n = len(member_columns[0])
    if n == 0:
        raise ValueError("members must contain at least one prediction")
    for col in member_columns:
        if len(col) != n:
            raise ValueError("all ensemble members must have equal length")
    out: List[float] = []
    for i in range(n):
        out.append(ensemble_mean([col[i] for col in member_columns]))
    return out


def evaluate_ensemble(
    member_columns: Sequence[Sequence[float]],
    actual: Sequence[float],
) -> Tuple[float, float]:
    """Return ``(rmse, mape)`` of the averaged ensemble against ``actual``."""
    preds = ensemble_predict(member_columns)
    return rmse(preds, actual), mape(preds, actual)
