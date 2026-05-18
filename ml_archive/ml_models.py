"""
ml_models.py — Implementaciones de los 5 métodos de ML para los robots.

Métodos implementados:
1. Linear Regression (OLS) — ANYmal: predice tiempo restante al objetivo
2. Logistic Regression — Husky: clasifica si la maniobra es segura
3. K-Means — PuzzleBot: descubre zonas operacionales sin etiquetas
4. Ridge Regression — Coordinator: estabiliza coeficientes con features correlacionadas
5. Random Forest — Benchmark no lineal (disponible si se necesita)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ===========================================================================
# 1. LINEAR REGRESSION (OLS) — ANYmal Time-to-Goal Predictor
# ===========================================================================

class LinearRegressionOLS:
    """Regresión lineal por mínimos cuadrados ordinarios.
    
    Uso: Predecir tiempo restante al objetivo para ANYmal.
    Features: [distance_to_goal, current_velocity, payload_kg, avg_det_J]
    Target: time_remaining (segundos)
    """
    
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.weights = None  # θ = (X^T X)^-1 X^T y
        self.bias = 0.0
        self.trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena el modelo usando OLS.
        
        Args:
            X: (n_samples, n_features) - matriz de features
            y: (n_samples,) - vector de targets
        """
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]  # Agregar columna de 1s para bias
        
        # θ = (X^T X)^-1 X^T y
        XtX = X_bias.T @ X_bias
        Xty = X_bias.T @ y
        theta = np.linalg.solve(XtX, Xty)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        self.trained = True
        
        # Calcular R²
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice valores para nuevas muestras."""
        if not self.trained:
            raise ValueError("Modelo no entrenado. Llama a fit() primero.")
        return X @ self.weights + self.bias
    
    def get_coefficients(self) -> Dict[str, float]:
        """Retorna los coeficientes del modelo."""
        return {
            "bias": float(self.bias),
            "weights": self.weights.tolist() if self.weights is not None else []
        }


# ===========================================================================
# 2. LOGISTIC REGRESSION — Husky Collision Safety Classifier
# ===========================================================================

class LogisticRegression:
    """Regresión logística para clasificación binaria.
    
    Uso: Clasificar si una maniobra del Husky es segura (1) o no (0).
    Features: [min_lidar_range, avg_lidar_range, angle_to_box, distance_to_box, 
               box_in_corridor, velocity]
    Target: is_safe (0 o 1)
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.n_features = n_features
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.trained = False
        
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Función sigmoide: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena el modelo usando gradient descent.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples,) - etiquetas binarias {0, 1}
        """
        n_samples = X.shape[0]
        
        for i in range(self.n_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)
            
            # Gradientes
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Log loss cada 100 iteraciones
            if i % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
                if i % 500 == 0:
                    print(f"  [LogisticRegression] Iter {i}: Loss={loss:.4f}")
        
        self.trained = True
        
        # Calcular accuracy
        y_pred_class = self.predict(X)
        accuracy = np.mean(y_pred_class == y)
        return accuracy
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades P(y=1|X)."""
        if not self.trained:
            raise ValueError("Modelo no entrenado.")
        z = X @ self.weights + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna clases predichas {0, 1}."""
        return (self.predict_proba(X) >= 0.5).astype(int)


# ===========================================================================
# 3. K-MEANS CLUSTERING — PuzzleBot Workspace Zone Discovery
# ===========================================================================

class KMeans:
    """K-Means clustering para descubrir zonas operacionales.
    
    Uso: Descubrir zonas en el workspace del PuzzleBot sin etiquetas.
    Features: [x_position, y_position, arm_extension, task_frequency]
    Output: cluster_labels, cluster_centers
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, tol: float = 1e-4):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.labels = None
        self.inertia = None
        self.trained = False
        
    def fit(self, X: np.ndarray):
        """Entrena K-Means usando Lloyd's algorithm.
        
        Args:
            X: (n_samples, n_features) - datos a agrupar
        """
        n_samples, n_features = X.shape
        
        # Inicialización: seleccionar k puntos aleatorios como centros
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centers = X[idx].copy()
        
        for iteration in range(self.max_iter):
            # Asignar cada punto al centro más cercano
            distances = np.zeros((n_samples, self.k))
            for i in range(self.k):
                distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # Actualizar centros
            new_centers = np.zeros((self.k, n_features))
            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centers[i] = cluster_points.mean(axis=0)
                else:
                    new_centers[i] = self.centers[i]  # Mantener centro si cluster vacío
            
            # Verificar convergencia
            center_shift = np.linalg.norm(new_centers - self.centers)
            self.centers = new_centers
            
            if center_shift < self.tol:
                print(f"  [KMeans] Convergió en {iteration + 1} iteraciones")
                break
        
        self.labels = labels
        
        # Calcular inertia (suma de distancias cuadradas a centros)
        self.inertia = 0.0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                self.inertia += np.sum((cluster_points - self.centers[i]) ** 2)
        
        self.trained = True
        return self.labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Asigna nuevos puntos a clusters existentes."""
        if not self.trained:
            raise ValueError("Modelo no entrenado.")
        
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.k))
        for i in range(self.k):
            distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
        
        return np.argmin(distances, axis=1)
    
    def get_cluster_info(self) -> Dict:
        """Retorna información de los clusters."""
        return {
            "n_clusters": self.k,
            "centers": self.centers.tolist() if self.centers is not None else [],
            "inertia": float(self.inertia) if self.inertia is not None else 0.0
        }


# ===========================================================================
# 4. RIDGE REGRESSION — Coordinator Mission Outcome Predictor
# ===========================================================================

class RidgeRegression:
    """Ridge Regression (L2 regularization) para features correlacionadas.
    
    Uso: Predecir tiempo total de misión o probabilidad de éxito del Coordinator.
    Features: [phase1_time, phase2_time, phase3_time, husky_slip, anymal_det_J_min,
               puzzlebot_stack_height, xarm_success]
    Target: total_mission_time o success_probability
    
    Regularización: θ = (X^T X + λI)^-1 X^T y
    """
    
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha  # Parámetro de regularización λ
        self.weights = None
        self.bias = 0.0
        self.trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena Ridge Regression.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples,) - targets
        """
        n_samples = X.shape[0]
        X_bias = np.c_[np.ones(n_samples), X]
        
        # θ = (X^T X + λI)^-1 X^T y
        XtX = X_bias.T @ X_bias
        # Agregar regularización (no regularizar el bias)
        reg_matrix = self.alpha * np.eye(X_bias.shape[1])
        reg_matrix[0, 0] = 0  # No regularizar bias
        
        XtX_reg = XtX + reg_matrix
        Xty = X_bias.T @ y
        theta = np.linalg.solve(XtX_reg, Xty)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        self.trained = True
        
        # Calcular R²
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice valores."""
        if not self.trained:
            raise ValueError("Modelo no entrenado.")
        return X @ self.weights + self.bias
    
    def get_coefficients(self) -> Dict[str, float]:
        """Retorna coeficientes."""
        return {
            "bias": float(self.bias),
            "weights": self.weights.tolist() if self.weights is not None else [],
            "alpha": self.alpha
        }


# ===========================================================================
# 5. RANDOM FOREST — Nonlinear Benchmark (Simplified)
# ===========================================================================

class DecisionTreeRegressor:
    """Árbol de decisión simple para regresión (usado por Random Forest)."""
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def _mse(self, y: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.var(y) * len(y)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Encuentra el mejor split (feature, threshold)."""
        best_mse = float('inf')
        best_feature = 0
        best_threshold = 0.0
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                
                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                mse_total = mse_left + mse_right
                
                if mse_total < best_mse:
                    best_mse = mse_total
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Construye el árbol recursivamente."""
        n_samples = len(y)
        
        # Condiciones de parada
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return {"value": np.mean(y)}
        
        feature, threshold = self._best_split(X, y)
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {"value": np.mean(y)}
        
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena el árbol."""
        self.tree = self._build_tree(X, y)
    
    def _predict_sample(self, x: np.ndarray, tree: Dict) -> float:
        """Predice un solo sample."""
        if "value" in tree:
            return tree["value"]
        
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice múltiples samples."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestRegressor:
    """Random Forest para regresión (benchmark no lineal).
    
    Uso: Cuando los modelos lineales no son suficientes.
    """
    
    def __init__(self, n_estimators: int = 10, max_depth: int = 5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees: List[DecisionTreeRegressor] = []
        self.trained = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Entrena el bosque con bootstrap sampling."""
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            # Entrenar árbol
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        self.trained = True
        
        # Calcular R² en datos de entrenamiento
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return r2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice promediando todos los árboles."""
        if not self.trained:
            raise ValueError("Modelo no entrenado.")
        
        predictions = np.zeros((len(self.trees), X.shape[0]))
        for i, tree in enumerate(self.trees):
            predictions[i] = tree.predict(X)
        
        return np.mean(predictions, axis=0)


# ===========================================================================
# UTILIDADES
# ===========================================================================

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza features a media 0 y std 1.
    
    Returns:
        X_norm, mean, std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-9  # Evitar división por 0
    X_norm = (X - mean) / std
    return X_norm, mean, std


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
    """Split simple train/test."""
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    idx = np.random.permutation(n_samples)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
