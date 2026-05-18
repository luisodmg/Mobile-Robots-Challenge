"""
ml_coordinator_predictor.py — Integración de Ridge Regression para Coordinator.

Predictor de resultados de misión con features correlacionadas:
- Input: métricas de las 3 fases (tiempos, errores, éxitos)
- Output: tiempo total de misión o probabilidad de éxito
"""

import numpy as np
from typing import Tuple, Dict
from ml_models import RidgeRegression, normalize_features, train_test_split


class CoordinatorMissionPredictor:
    """Predictor ML de resultados de misión para el Coordinator."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: parámetro de regularización L2 (mayor = más regularización)
        """
        self.model = RidgeRegression(n_features=7, alpha=alpha)
        self.feature_mean = None
        self.feature_std = None
        self.trained = False
        
    def extract_features(
        self,
        phase1_time: float,
        phase2_time: float,
        phase3_time: float,
        husky_slip: float,
        anymal_det_J_min: float,
        puzzlebot_stack_height: float,
        xarm_success: int
    ) -> np.ndarray:
        """Extrae features de las métricas de la misión.
        
        Features (muchas correlacionadas entre sí):
        1. phase1_time: tiempo de Fase 1 [s]
        2. phase2_time: tiempo de Fase 2 [s]
        3. phase3_time: tiempo de Fase 3 [s]
        4. husky_slip: factor de deslizamiento del Husky [0-1]
        5. anymal_det_J_min: mínimo det(J) del ANYmal durante trayecto
        6. puzzlebot_stack_height: altura final de la pila [m]
        7. xarm_success: éxito de XArm {0, 1}
        """
        features = np.array([
            phase1_time,
            phase2_time,
            phase3_time,
            husky_slip,
            anymal_det_J_min,
            puzzlebot_stack_height,
            float(xarm_success)
        ])
        return features
    
    def generate_training_data(
        self,
        n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos sintéticos de misiones simuladas.
        
        Target: total_mission_time
        
        Modelo con correlaciones:
        - phase1_time, phase2_time, phase3_time están correlacionadas con total_time
        - husky_slip afecta phase1_time
        - anymal_det_J_min afecta phase2_time
        - puzzlebot_stack_height y xarm_success afectan phase3_time
        """
        X = []
        y = []
        
        for _ in range(n_samples):
            # Parámetros base aleatorios
            husky_slip = np.random.uniform(0.0, 0.15)
            anymal_det_J_min = np.random.uniform(0.005, 0.3)
            xarm_success = np.random.choice([0, 1], p=[0.1, 0.9])  # 90% éxito
            
            # Fase 1: tiempo base + efecto del slip
            phase1_base = np.random.uniform(15.0, 25.0)
            phase1_time = phase1_base + husky_slip * 20.0  # Slip aumenta tiempo
            
            # Fase 2: tiempo base + efecto de det(J) bajo
            phase2_base = np.random.uniform(20.0, 35.0)
            if anymal_det_J_min < 0.05:
                phase2_penalty = 10.0  # Singularidades ralentizan
            else:
                phase2_penalty = 0.0
            phase2_time = phase2_base + phase2_penalty
            
            # Fase 3: tiempo base + efecto de XArm
            phase3_base = np.random.uniform(25.0, 40.0)
            if xarm_success == 0:
                phase3_penalty = 15.0  # Fallo de XArm requiere reposicionamiento manual
            else:
                phase3_penalty = 0.0
            phase3_time = phase3_base + phase3_penalty
            
            # Stack height (correlacionado con phase3_time)
            if phase3_time < 30:
                stack_height = np.random.uniform(0.14, 0.15)  # Pila completa
            else:
                stack_height = np.random.uniform(0.05, 0.12)  # Pila incompleta
            
            # Total time (suma + overhead + ruido)
            overhead = np.random.uniform(5.0, 10.0)  # Transiciones entre fases
            total_time = phase1_time + phase2_time + phase3_time + overhead
            total_time += np.random.normal(0, 2.0)  # Ruido
            
            features = np.array([
                phase1_time,
                phase2_time,
                phase3_time,
                husky_slip,
                anymal_det_J_min,
                stack_height,
                float(xarm_success)
            ])
            
            X.append(features)
            y.append(total_time)
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Entrena Ridge Regression con regularización L2."""
        print(f"\n[CoordinatorMissionPredictor] Entrenando Ridge Regression (α={self.model.alpha})...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Normalizar features
        X_train_norm, self.feature_mean, self.feature_std = normalize_features(X_train)
        X_test_norm = (X_test - self.feature_mean) / self.feature_std
        
        # Entrenar
        r2_train = self.model.fit(X_train_norm, y_train)
        print(f"  R² en train: {r2_train:.3f}")
        
        # Evaluar en test
        y_pred_test = self.model.predict(X_test_norm)
        ss_res = np.sum((y_test - y_pred_test) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_test = 1 - (ss_res / ss_tot)
        
        mae_test = np.mean(np.abs(y_test - y_pred_test))
        rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        
        print(f"  R² en test: {r2_test:.3f}")
        print(f"  MAE en test: {mae_test:.3f} s")
        print(f"  RMSE en test: {rmse_test:.3f} s")
        
        self.trained = True
        return r2_test
    
    def predict_total_time(
        self,
        phase1_time: float,
        phase2_time: float,
        phase3_time: float,
        husky_slip: float,
        anymal_det_J_min: float,
        puzzlebot_stack_height: float,
        xarm_success: int
    ) -> float:
        """Predice el tiempo total de misión.
        
        Returns:
            tiempo total estimado [segundos]
        """
        if not self.trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        features = self.extract_features(
            phase1_time, phase2_time, phase3_time,
            husky_slip, anymal_det_J_min, puzzlebot_stack_height, xarm_success
        )
        
        features_norm = (features - self.feature_mean) / self.feature_std
        features_norm = features_norm.reshape(1, -1)
        
        time_pred = self.model.predict(features_norm)[0]
        return max(time_pred, 0.0)
    
    def get_coefficients(self) -> Dict[str, float]:
        """Retorna los coeficientes regularizados del modelo."""
        if not self.trained:
            return {}
        
        feature_names = [
            "phase1_time",
            "phase2_time",
            "phase3_time",
            "husky_slip",
            "anymal_det_J_min",
            "puzzlebot_stack_height",
            "xarm_success"
        ]
        
        coeffs = self.model.get_coefficients()
        weights = coeffs["weights"]
        
        result = {
            "bias": coeffs["bias"],
            "alpha": coeffs["alpha"]
        }
        for name, w in zip(feature_names, weights):
            result[name] = float(w)
        
        return result
    
    def compare_with_ols(self, X: np.ndarray, y: np.ndarray):
        """Compara Ridge vs OLS para mostrar efecto de regularización."""
        from ml_models import LinearRegressionOLS
        
        print("\n[Comparación Ridge vs OLS]")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_norm, mean, std = normalize_features(X_train)
        X_test_norm = (X_test - mean) / std
        
        # Ridge (ya entrenado)
        y_pred_ridge = self.model.predict(X_test_norm)
        r2_ridge = 1 - np.sum((y_test - y_pred_ridge)**2) / np.sum((y_test - np.mean(y_test))**2)
        
        # OLS
        ols = LinearRegressionOLS(n_features=7)
        ols.fit(X_train_norm, y_train)
        y_pred_ols = ols.predict(X_test_norm)
        r2_ols = 1 - np.sum((y_test - y_pred_ols)**2) / np.sum((y_test - np.mean(y_test))**2)
        
        print(f"  Ridge R² (α={self.model.alpha}): {r2_ridge:.4f}")
        print(f"  OLS R²:                          {r2_ols:.4f}")
        
        # Magnitud de coeficientes (Ridge debería tener menores)
        ridge_weights = self.model.weights
        ols_weights = ols.weights
        
        print(f"\n  Magnitud de coeficientes:")
        print(f"    Ridge: {np.linalg.norm(ridge_weights):.4f}")
        print(f"    OLS:   {np.linalg.norm(ols_weights):.4f}")
        print(f"    Reducción: {(1 - np.linalg.norm(ridge_weights)/np.linalg.norm(ols_weights))*100:.1f}%")
        
        return r2_ridge, r2_ols
    
    def analyze_feature_correlation(self, X: np.ndarray):
        """Analiza la correlación entre features."""
        print("\n[Análisis de Correlación de Features]")
        
        feature_names = [
            "phase1_time", "phase2_time", "phase3_time",
            "husky_slip", "anymal_det_J_min", "stack_height", "xarm_success"
        ]
        
        # Matriz de correlación
        corr_matrix = np.corrcoef(X.T)
        
        print("  Correlaciones altas (|r| > 0.5):")
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.5:
                    print(f"    {feature_names[i]:25s} ↔ {feature_names[j]:25s}: r={corr:+.3f}")
        
        print("\n  → Ridge Regression maneja estas correlaciones con regularización L2")


if __name__ == "__main__":
    print("=" * 60)
    print("  Test: Coordinator Mission Predictor (Ridge Regression)")
    print("=" * 60)
    
    np.random.seed(42)
    
    predictor = CoordinatorMissionPredictor(alpha=1.0)
    
    # Generar datos de entrenamiento
    print("\n[1] Generando datos de misiones simuladas...")
    X, y = predictor.generate_training_data(n_samples=800)
    print(f"  Generadas {len(X)} misiones")
    print(f"  Rango de tiempos totales: [{y.min():.1f}, {y.max():.1f}] s")
    print(f"  Tiempo promedio: {y.mean():.1f} s")
    
    # Analizar correlaciones
    predictor.analyze_feature_correlation(X)
    
    # Entrenar
    print("\n[2] Entrenando predictor...")
    r2_test = predictor.train(X, y, test_size=0.2)
    
    # Comparar con OLS
    print("\n[3] Comparando Ridge vs OLS...")
    predictor.compare_with_ols(X, y)
    
    # Coeficientes
    print("\n[4] Coeficientes del modelo Ridge:")
    coeffs = predictor.get_coefficients()
    print(f"  Bias: {coeffs['bias']:.4f}")
    print(f"  Alpha (regularización): {coeffs['alpha']:.4f}")
    print("\n  Pesos de features:")
    for name in ["phase1_time", "phase2_time", "phase3_time", "husky_slip",
                 "anymal_det_J_min", "puzzlebot_stack_height", "xarm_success"]:
        print(f"    {name:25s}: {coeffs[name]:+.4f}")
    
    # Test en escenarios específicos
    print("\n[5] Probando escenarios específicos:")
    
    # Escenario 1: Misión rápida y exitosa
    time_pred = predictor.predict_total_time(
        phase1_time=18.0,
        phase2_time=22.0,
        phase3_time=28.0,
        husky_slip=0.02,
        anymal_det_J_min=0.15,
        puzzlebot_stack_height=0.15,
        xarm_success=1
    )
    print(f"  Escenario ÓPTIMO: {time_pred:.1f} s")
    
    # Escenario 2: Misión con problemas
    time_pred = predictor.predict_total_time(
        phase1_time=28.0,
        phase2_time=42.0,
        phase3_time=50.0,
        husky_slip=0.12,
        anymal_det_J_min=0.02,
        puzzlebot_stack_height=0.08,
        xarm_success=0
    )
    print(f"  Escenario PROBLEMÁTICO: {time_pred:.1f} s")
    
    # Escenario 3: Misión promedio
    time_pred = predictor.predict_total_time(
        phase1_time=20.0,
        phase2_time=28.0,
        phase3_time=32.0,
        husky_slip=0.05,
        anymal_det_J_min=0.08,
        puzzlebot_stack_height=0.12,
        xarm_success=1
    )
    print(f"  Escenario PROMEDIO: {time_pred:.1f} s")
    
    print("\n[OK] Test completado.")
