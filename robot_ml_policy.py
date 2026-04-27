"""
robot_ml_policy.py - Lightweight random-forest policy for robot adaptation.

The policy learns three binary classifiers from synthetic examples generated from the
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


@dataclass
class _TreeNode:
    """Node for a binary classification tree."""

    prob: float
    feature_idx: int = -1
    threshold: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


@dataclass
class _SimpleDecisionTree:
    """Small CART-like tree for binary classification with Gini splitting."""

    max_depth: int
    min_samples_split: int
    max_features: int
    root: _TreeNode | None = None

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
        max_depth: int = 5,
        min_samples_split: int = 12,
        max_features: int | None = None,
    ) -> "_SimpleDecisionTree":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if max_features is None:
            max_features = max(1, int(np.sqrt(X.shape[1])))

        model = cls(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
        )
        idx = np.arange(X.shape[0])
        model.root = model._build(X, y, idx, depth=0, rng=rng)
        return model

    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        p = float(np.mean(y))
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idx: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, float] | tuple[None, None]:
        n_features = X.shape[1]
        feat_pool = rng.choice(n_features, size=min(self.max_features, n_features), replace=False)

        best_feat = None
        best_thr = None
        best_cost = np.inf

        for feat in feat_pool:
            vals = X[idx, feat]
            uniq = np.unique(vals)
            if uniq.size < 2:
                continue

            if uniq.size > 20:
                q = np.linspace(0.05, 0.95, 15)
                thresholds = np.quantile(vals, q)
            else:
                thresholds = (uniq[:-1] + uniq[1:]) * 0.5

            for thr in thresholds:
                left_idx = idx[X[idx, feat] <= thr]
                right_idx = idx[X[idx, feat] > thr]
                if left_idx.size == 0 or right_idx.size == 0:
                    continue

                g_left = self._gini(y[left_idx])
                g_right = self._gini(y[right_idx])
                cost = (left_idx.size * g_left + right_idx.size * g_right) / idx.size

                if cost < best_cost:
                    best_cost = cost
                    best_feat = int(feat)
                    best_thr = float(thr)

        return best_feat, best_thr

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        idx: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> _TreeNode:
        prob = float(np.mean(y[idx])) if idx.size > 0 else 0.5
        node = _TreeNode(prob=prob)

        if (
            depth >= self.max_depth
            or idx.size < self.min_samples_split
            or prob <= 1e-6
            or prob >= 1.0 - 1e-6
        ):
            return node

        feat, thr = self._best_split(X, y, idx, rng)
        if feat is None:
            return node

        left_idx = idx[X[idx, feat] <= thr]
        right_idx = idx[X[idx, feat] > thr]
        if left_idx.size == 0 or right_idx.size == 0:
            return node

        node.feature_idx = feat
        node.threshold = thr
        node.left = self._build(X, y, left_idx, depth + 1, rng)
        node.right = self._build(X, y, right_idx, depth + 1, rng)
        return node

    def predict_proba(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        node = self.root
        if node is None:
            return 0.5

        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left if node.left is not None else node
            else:
                node = node.right if node.right is not None else node
            if node.left is None or node.right is None:
                break

        return float(node.prob)


@dataclass
class RandomForestModel:
    """Small random-forest classifier for binary probability outputs."""

    trees: list[_SimpleDecisionTree]

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        n_trees: int = 25,
        max_depth: int = 5,
        min_samples_split: int = 12,
        max_features: int | None = None,
        bootstrap_ratio: float = 1.0,
        seed: int = 42,
    ) -> "RandomForestModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        rng = np.random.default_rng(seed)

        trees: list[_SimpleDecisionTree] = []
        n = X.shape[0]
        n_boot = max(16, int(n * bootstrap_ratio))

        for _ in range(n_trees):
            boot_idx = rng.integers(0, n, size=n_boot)
            tree_rng = np.random.default_rng(int(rng.integers(0, 1_000_000_000)))
            tree = _SimpleDecisionTree.fit(
                X[boot_idx],
                y[boot_idx],
                rng=tree_rng,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
            )
            trees.append(tree)

        return cls(trees=trees)

    def predict_proba(self, x: np.ndarray) -> float:
        if not self.trees:
            return 0.5
        probs = [tree.predict_proba(x) for tree in self.trees]
        return float(np.mean(probs))


@dataclass
class RobotMLPolicy:
    """Shared ML policy used by all robots in the pipeline."""

    husky_model: RandomForestModel
    anymal_model: RandomForestModel
    arm_model: RandomForestModel

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
            husky_model=RandomForestModel.fit(
                np.array(husky_X),
                np.array(husky_y),
                n_trees=27,
                max_depth=5,
                min_samples_split=10,
                seed=seed + 1,
            ),
            anymal_model=RandomForestModel.fit(
                np.array(anymal_X),
                np.array(anymal_y),
                n_trees=27,
                max_depth=5,
                min_samples_split=10,
                seed=seed + 2,
            ),
            arm_model=RandomForestModel.fit(
                np.array(arm_X),
                np.array(arm_y),
                n_trees=27,
                max_depth=5,
                min_samples_split=10,
                seed=seed + 3,
            ),
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