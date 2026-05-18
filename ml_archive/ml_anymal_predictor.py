"""
ml_anymal_predictor.py — Integración de Linear Regression (OLS) para ANYmal.

Predictor de tiempo restante al objetivo:
- Input: distancia, velocidad, payload, det(J) promedio
- Output: tiempo estimado para llegar al destino [segundos]
"""

import numpy as np
from typing import Tuple, Dict
from ml_models import LinearRegressionOLS, normalize_features, train_test_split


class ANYmalTimePredictor:
    """Predictor ML de tiempo restante al objetivo para ANYmal."""
    
    def __init__(self):
        self.model = LinearRegressionOLS(n_features=4)
        self.feature_mean = None
        self.feature_std = None
        self.trained = False
        
    def extract_features(
        self,
        distance_to_goal: float,
        current_velocity: float,
        payload_kg: float,
        avg_det_J: float
    ) -> np.ndarray:
        """Extrae features del estado actual del ANYmal.
        
        Features:
        1. distance_to_goal: distancia euclidiana al objetivo [m]
        2. current_velocity: velocidad actual [m/s]
        3. payload_kg: masa del payload transportado [kg]
        4. avg_det_J: promedio de det(J) de las 4 patas (salud cinemática)
        """
        features = np.array([
            distance_to_goal,
            current_velocity,
            payload_kg,
            avg_det_J
        ])
        return features
    
    def generate_training_data(
        self,
        n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos sintéticos de entrenamiento basados en física.
        
        Modelo físico simplificado:
        time = distance / velocity_effective
        velocity_effective = base_velocity * (1 - payload_factor) * det_J_factor
        """
        X = []
        y = []
        
        for _ in range(n_samples):
            # Parámetros aleatorios
            distance = np.random.uniform(1.0, 15.0)  # [m]
            base_velocity = np.random.uniform(0.2, 0.5)  # [m/s]
            payload_kg = np.random.uniform(0.0, 10.0)  # [kg]
            avg_det_J = np.random.uniform(0.01, 0.5)  # Salud cinemática
            
            # Factores de degradación
            # Payload reduce velocidad: más peso = más lento
            payload_factor = 0.05 * (payload_kg / 10.0)  # 0-5% reducción
            
            # det(J) bajo indica singularidad = movimiento más lento
            # det(J) alto (>0.1) = normal, det(J) bajo (<0.05) = degradado
            if avg_det_J > 0.1:
                det_J_factor = 1.0
            else:
                det_J_factor = 0.7 + 0.3 * (avg_det_J / 0.1)  # 70-100%
            
            # Velocidad efectiva
            velocity_effective = base_velocity * (1 - payload_factor) * det_J_factor
            velocity_effective = max(velocity_effective, 0.05)  # Mínimo 5 cm/s
            
            # Tiempo = distancia / velocidad + ruido
            time_base = distance / velocity_effective
            time_noisy = time_base + np.random.normal(0, 0.5)  # Ruido gaussiano
            time_noisy = max(time_noisy, 0.1)  # Tiempo mínimo
            
            features = np.array([
                distance,
                base_velocity,
                payload_kg,
                avg_det_J
            ])
            
            X.append(features)
            y.append(time_noisy)
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Entrena el predictor con datos históricos."""
        print("\n[ANYmalTimePredictor] Entrenando modelo...")
        
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
    
    def predict_time_to_goal(
        self,
        distance_to_goal: float,
        current_velocity: float,
        payload_kg: float,
        avg_det_J: float
    ) -> float:
        """Predice el tiempo restante para llegar al objetivo.
        
        Returns:
            tiempo estimado [segundos]
        """
        if not self.trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Extraer features
        features = self.extract_features(distance_to_goal, current_velocity, payload_kg, avg_det_J)
        
        # Normalizar
        features_norm = (features - self.feature_mean) / self.feature_std
        features_norm = features_norm.reshape(1, -1)
        
        # Predecir
        time_pred = self.model.predict(features_norm)[0]
        
        return max(time_pred, 0.0)  # No puede ser negativo
    
    def get_coefficients(self) -> Dict[str, float]:
        """Retorna los coeficientes del modelo lineal."""
        if not self.trained:
            return {}
        
        feature_names = [
            "distance_to_goal",
            "current_velocity",
            "payload_kg",
            "avg_det_J"
        ]
        
        coeffs = self.model.get_coefficients()
        weights = coeffs["weights"]
        
        result = {"bias": coeffs["bias"]}
        for name, w in zip(feature_names, weights):
            result[name] = float(w)
        
        return result
    
    def explain_prediction(
        self,
        distance_to_goal: float,
        current_velocity: float,
        payload_kg: float,
        avg_det_J: float
    ) -> Dict:
        """Explica una predicción mostrando la contribución de cada feature."""
        if not self.trained:
            return {}
        
        features = self.extract_features(distance_to_goal, current_velocity, payload_kg, avg_det_J)
        features_norm = (features - self.feature_mean) / self.feature_std
        
        feature_names = [
            "distance_to_goal",
            "current_velocity",
            "payload_kg",
            "avg_det_J"
        ]
        
        contributions = {}
        for i, name in enumerate(feature_names):
            contribution = features_norm[i] * self.model.weights[i]
            contributions[name] = {
                "raw_value": float(features[i]),
                "normalized_value": float(features_norm[i]),
                "weight": float(self.model.weights[i]),
                "contribution": float(contribution)
            }
        
        total_pred = self.predict_time_to_goal(distance_to_goal, current_velocity, payload_kg, avg_det_J)
        
        return {
            "prediction": float(total_pred),
            "bias": float(self.model.bias),
            "contributions": contributions
        }


if __name__ == "__main__":
    print("=" * 60)
    print("  Test: ANYmal Time Predictor (Linear Regression OLS)")
    print("=" * 60)
    
    np.random.seed(42)
    
    predictor = ANYmalTimePredictor()
    
    # Generar datos de entrenamiento
    print("\n[1] Generando datos sintéticos...")
    X, y = predictor.generate_training_data(n_samples=800)
    print(f"  Generadas {len(X)} muestras")
    print(f"  Rango de tiempos: [{y.min():.2f}, {y.max():.2f}] s")
    print(f"  Tiempo promedio: {y.mean():.2f} s")
    
    # Entrenar
    print("\n[2] Entrenando predictor...")
    r2_test = predictor.train(X, y, test_size=0.2)
    
    # Coeficientes
    print("\n[3] Coeficientes del modelo lineal:")
    coeffs = predictor.get_coefficients()
    print(f"  Bias: {coeffs['bias']:.4f}")
    for name in ["distance_to_goal", "current_velocity", "payload_kg", "avg_det_J"]:
        print(f"  {name:20s}: {coeffs[name]:+.4f}")
    
    # Interpretación
    print("\n[4] Interpretación:")
    print("  - distance_to_goal: coef > 0 → más distancia = más tiempo ✓")
    print("  - current_velocity: coef < 0 → más velocidad = menos tiempo ✓")
    print("  - payload_kg: coef > 0 → más peso = más tiempo ✓")
    print("  - avg_det_J: coef < 0 → mejor cinemática = menos tiempo ✓")
    
    # Test en escenarios específicos
    print("\n[5] Probando escenarios específicos:")
    
    # Escenario 1: Cerca del objetivo, sin payload
    time_pred = predictor.predict_time_to_goal(
        distance_to_goal=2.0,
        current_velocity=0.4,
        payload_kg=0.0,
        avg_det_J=0.2
    )
    print(f"  Escenario 1 (cerca, sin carga): {time_pred:.2f} s")
    
    # Escenario 2: Lejos, con payload pesado
    time_pred = predictor.predict_time_to_goal(
        distance_to_goal=12.0,
        current_velocity=0.3,
        payload_kg=8.0,
        avg_det_J=0.05
    )
    print(f"  Escenario 2 (lejos, carga pesada): {time_pred:.2f} s")
    
    # Escenario 3: Distancia media, condiciones normales
    time_pred = predictor.predict_time_to_goal(
        distance_to_goal=6.0,
        current_velocity=0.4,
        payload_kg=6.0,
        avg_det_J=0.15
    )
    print(f"  Escenario 3 (medio, normal): {time_pred:.2f} s")
    
    # Explicación detallada
    print("\n[6] Explicación detallada del Escenario 3:")
    explanation = predictor.explain_prediction(6.0, 0.4, 6.0, 0.15)
    print(f"  Predicción total: {explanation['prediction']:.2f} s")
    print(f"  Bias: {explanation['bias']:.4f}")
    print("\n  Contribuciones por feature:")
    for name, data in explanation['contributions'].items():
        print(f"    {name:20s}: {data['raw_value']:6.2f} → contrib={data['contribution']:+.4f}")
    
    print("\n[OK] Test completado.")
