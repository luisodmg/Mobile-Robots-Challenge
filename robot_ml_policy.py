"""
robot_ml_policy.py - Lightweight logistic-regression policy for robot adaptation.

The policy learns three small classifiers from synthetic examples generated from the
existing rule-based behavior:
- Husky navigation safety
- ANYmal gait speed safety
- PuzzleBot arm contact safety

The models are used as advisory signals to scale speed and contact force while
keeping the rule-based controllers as the primary control loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class LogisticModel:
    """Small logistic-regression model trained with gradient descent."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.15,
        steps: int = 1800,
        l2: float = 0.01,
    ) -> "LogisticModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        Xn = (X - mean) / scale

        weights = np.zeros(X.shape[1], dtype=float)
        bias = 0.0

        n = Xn.shape[0]
        for _ in range(steps):
            logits = Xn @ weights + bias
            pred = _sigmoid(logits)
            error = pred - y
            grad_w = (Xn.T @ error) / n + l2 * weights
            grad_b = float(error.mean())
            weights -= lr * grad_w
            bias -= lr * grad_b

        return cls(weights=weights, bias=bias, mean=mean, scale=scale)

    def predict_proba(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        xn = (x - self.mean) / self.scale
        return float(_sigmoid(xn @ self.weights + self.bias))


@dataclass
class RobotMLPolicy:
    """Shared ML policy used by all robots in the pipeline."""

    husky_model: LogisticModel
    anymal_model: LogisticModel
    arm_model: LogisticModel

    @classmethod
    def build_default(cls, seed: int = 42) -> "RobotMLPolicy":
        rng = np.random.default_rng(seed)

        husky_X = []
        husky_y = []
        for _ in range(1200):
            dist = rng.uniform(0.0, 4.0)
            angle_err = rng.uniform(0.0, np.pi)
            nearest_box = rng.uniform(0.08, 2.0)
            pushing = float(rng.integers(0, 2))

            husky_X.append([dist, angle_err, nearest_box, pushing])
            safe = (
                angle_err < (0.65 - 0.10 * pushing)
                and nearest_box > (0.35 + 0.15 * pushing)
            ) or dist < 0.2
            husky_y.append(float(safe))

        anymal_X = []
        anymal_y = []
        for _ in range(1200):
            dist = rng.uniform(0.0, 8.0)
            det_j = 10 ** rng.uniform(-4.5, -0.1)
            payload = rng.uniform(4.0, 8.0)

            anymal_X.append([dist, det_j, payload])
            safe = (det_j > 1.2e-3 and dist < 6.0) or (det_j > 3e-3)
            anymal_y.append(float(safe))

        arm_X = []
        arm_y = []
        for _ in range(1200):
            workspace_margin = rng.uniform(-0.08, 0.20)
            det_j = 10 ** rng.uniform(-4.5, -0.2)
            target_height = rng.uniform(0.0, 0.20)

            arm_X.append([workspace_margin, det_j, target_height])
            safe = (workspace_margin > 0.01 and det_j > 1.0e-3) or workspace_margin > 0.05
            arm_y.append(float(safe))

        return cls(
            husky_model=LogisticModel.fit(np.array(husky_X), np.array(husky_y)),
            anymal_model=LogisticModel.fit(np.array(anymal_X), np.array(anymal_y)),
            arm_model=LogisticModel.fit(np.array(arm_X), np.array(arm_y)),
        )

    def husky_navigation_profile(
        self,
        dist: float,
        angle_err: float,
        nearest_box_dist: float,
        pushing: bool,
    ) -> Tuple[float, float]:
        score = self.husky_model.predict_proba(
            np.array([dist, abs(angle_err), nearest_box_dist, float(pushing)])
        )
        speed_scale = 0.35 + 0.65 * score
        turn_gain = 1.8 + 0.9 * (1.0 - score)
        return speed_scale, turn_gain

    def anymal_speed_profile(self, dist: float, det_j_min: float, payload_kg: float) -> Tuple[float, float]:
        score = self.anymal_model.predict_proba(np.array([dist, det_j_min, payload_kg]))
        speed_scale = 0.30 + 0.70 * score
        step_gain = 0.85 + 0.25 * score
        return speed_scale, step_gain

    def arm_contact_profile(
        self,
        workspace_margin: float,
        det_j: float,
        target_height: float,
    ) -> Tuple[float, float]:
        score = self.arm_model.predict_proba(np.array([workspace_margin, det_j, target_height]))
        grip_scale = 0.90 + 0.20 * (1.0 - score)
        descent_scale = 0.75 + 0.25 * score
        return grip_scale, descent_scale


_POLICY: RobotMLPolicy | None = None


def get_robot_policy() -> RobotMLPolicy:
    global _POLICY
    if _POLICY is None:
        _POLICY = RobotMLPolicy.build_default()
    return _POLICY