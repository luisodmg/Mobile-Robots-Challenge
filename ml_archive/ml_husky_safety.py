"""
ml_husky_safety.py — Integración de Logistic Regression para Husky Pusher.

Clasificador de seguridad de maniobras:
- Input: LiDAR features, posición de cajas, velocidad
- Output: is_safe (1=seguro, 0=peligroso)
"""

import numpy as np
from typing import List, Dict, Tuple
from ml_models import LogisticRegression, normalize_features, train_test_split


class HuskySafetyClassifier:
    """Clasificador ML para determinar si una maniobra del Husky es segura."""
    
    def __init__(self):
        self.model = LogisticRegression(n_features=6, learning_rate=0.01, n_iterations=1000)
        self.feature_mean = None
        self.feature_std = None
        self.trained = False
        
    def extract_features(
        self,
        lidar_ranges: np.ndarray,
        robot_pos: np.ndarray,
        robot_theta: float,
        box_pos: np.ndarray,
        velocity: float
    ) -> np.ndarray:
        """Extrae features de la situación actual del Husky.
        
        Features:
        1. min_lidar_range: distancia mínima detectada por LiDAR
        2. avg_lidar_range: distancia promedio
        3. std_lidar_range: desviación estándar (variabilidad del entorno)
        4. angle_to_box: ángulo relativo a la caja objetivo
        5. distance_to_box: distancia euclidiana a la caja
        6. velocity: velocidad actual del robot
        """
        # LiDAR features
        min_range = np.min(lidar_ranges)
        avg_range = np.mean(lidar_ranges)
        std_range = np.std(lidar_ranges)
        
        # Box features
        dx = box_pos[0] - robot_pos[0]
        dy = box_pos[1] - robot_pos[1]
        distance_to_box = np.hypot(dx, dy)
        angle_to_box = np.arctan2(dy, dx) - robot_theta
        angle_to_box = (angle_to_box + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
        
        features = np.array([
            min_range,
            avg_range,
            std_range,
            angle_to_box,
            distance_to_box,
            velocity
        ])
        
        return features
    
    def generate_training_data(
        self,
        n_samples: int = 500,
        corridor_bounds: Tuple[float, float, float, float] = (1.0, 7.0, -1.0, 1.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos sintéticos de entrenamiento.
        
        Simula diferentes escenarios:
        - Seguro: LiDAR despejado, ángulo alineado, distancia adecuada
        - Peligroso: Obstáculos cercanos, ángulo malo, velocidad alta cerca de obstáculos
        """
        X = []
        y = []
        
        x_min, x_max, y_min, y_max = corridor_bounds
        
        for _ in range(n_samples):
            # Posición aleatoria del robot
            robot_x = np.random.uniform(x_min, x_max)
            robot_y = np.random.uniform(y_min, y_max)
            robot_theta = np.random.uniform(-np.pi, np.pi)
            robot_pos = np.array([robot_x, robot_y])
            
            # Posición aleatoria de la caja
            box_x = np.random.uniform(x_min + 1, x_max - 1)
            box_y = np.random.uniform(y_min + 0.5, y_max - 0.5)
            box_pos = np.array([box_x, box_y])
            
            # Velocidad aleatoria
            velocity = np.random.uniform(0.0, 1.0)
            
            # Simular LiDAR (simplificado)
            distance_to_box = np.linalg.norm(box_pos - robot_pos)
            
            # Si hay obstáculo cerca, LiDAR detecta distancias cortas
            if distance_to_box < 1.5:
                min_range = np.random.uniform(0.3, distance_to_box)
                avg_range = np.random.uniform(min_range, distance_to_box + 1.0)
                std_range = np.random.uniform(0.1, 0.5)
            else:
                min_range = np.random.uniform(2.0, 8.0)
                avg_range = np.random.uniform(min_range, 8.0)
                std_range = np.random.uniform(0.2, 1.0)
            
            # Ángulo a la caja
            dx = box_pos[0] - robot_pos[0]
            dy = box_pos[1] - robot_pos[1]
            angle_to_box = np.arctan2(dy, dx) - robot_theta
            angle_to_box = (angle_to_box + np.pi) % (2 * np.pi) - np.pi
            
            features = np.array([
                min_range,
                avg_range,
                std_range,
                angle_to_box,
                distance_to_box,
                velocity
            ])
            
            # Etiquetar como seguro (1) o peligroso (0)
            is_safe = 1
            
            # Condiciones de peligro
            if min_range < 0.5:  # Obstáculo muy cerca
                is_safe = 0
            elif distance_to_box < 0.8 and velocity > 0.5:  # Velocidad alta cerca de caja
                is_safe = 0
            elif abs(angle_to_box) > np.pi / 2 and velocity > 0.3:  # Ángulo malo con velocidad
                is_safe = 0
            elif std_range > 1.5:  # Entorno muy variable (caótico)
                is_safe = 0
            
            X.append(features)
            y.append(is_safe)
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Entrena el clasificador con datos etiquetados."""
        print("\n[HuskySafetyClassifier] Entrenando modelo...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Normalizar features
        X_train_norm, self.feature_mean, self.feature_std = normalize_features(X_train)
        X_test_norm = (X_test - self.feature_mean) / self.feature_std
        
        # Entrenar
        train_acc = self.model.fit(X_train_norm, y_train)
        print(f"  Accuracy en train: {train_acc:.3f}")
        
        # Evaluar en test
        y_pred_test = self.model.predict(X_test_norm)
        test_acc = np.mean(y_pred_test == y_test)
        print(f"  Accuracy en test: {test_acc:.3f}")
        
        # Matriz de confusión
        tp = np.sum((y_test == 1) & (y_pred_test == 1))
        tn = np.sum((y_test == 0) & (y_pred_test == 0))
        fp = np.sum((y_test == 0) & (y_pred_test == 1))
        fn = np.sum((y_test == 1) & (y_pred_test == 0))
        
        print(f"\n  Matriz de confusión:")
        print(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"    Precision: {precision:.3f}")
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"    Recall: {recall:.3f}")
        
        self.trained = True
        return test_acc
    
    def is_maneuver_safe(
        self,
        lidar_ranges: np.ndarray,
        robot_pos: np.ndarray,
        robot_theta: float,
        box_pos: np.ndarray,
        velocity: float
    ) -> Tuple[bool, float]:
        """Predice si la maniobra actual es segura.
        
        Returns:
            (is_safe, confidence) - bool y probabilidad [0,1]
        """
        if not self.trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Extraer features
        features = self.extract_features(lidar_ranges, robot_pos, robot_theta, box_pos, velocity)
        
        # Normalizar
        features_norm = (features - self.feature_mean) / self.feature_std
        features_norm = features_norm.reshape(1, -1)
        
        # Predecir
        prob = self.model.predict_proba(features_norm)[0]
        is_safe = prob >= 0.5
        
        return bool(is_safe), float(prob)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna la importancia de cada feature (magnitud de pesos)."""
        if not self.trained:
            return {}
        
        feature_names = [
            "min_lidar_range",
            "avg_lidar_range", 
            "std_lidar_range",
            "angle_to_box",
            "distance_to_box",
            "velocity"
        ]
        
        weights = np.abs(self.model.weights)
        importance = {name: float(w) for name, w in zip(feature_names, weights)}
        
        return importance


if __name__ == "__main__":
    print("=" * 60)
    print("  Test: Husky Safety Classifier (Logistic Regression)")
    print("=" * 60)
    
    np.random.seed(42)
    
    classifier = HuskySafetyClassifier()
    
    # Generar datos de entrenamiento
    print("\n[1] Generando datos sintéticos...")
    X, y = classifier.generate_training_data(n_samples=1000)
    print(f"  Generadas {len(X)} muestras")
    print(f"  Distribución de clases: Seguro={np.sum(y==1)}, Peligroso={np.sum(y==0)}")
    
    # Entrenar
    print("\n[2] Entrenando clasificador...")
    test_acc = classifier.train(X, y, test_size=0.2)
    
    # Feature importance
    print("\n[3] Importancia de features:")
    importance = classifier.get_feature_importance()
    for name, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {value:.4f}")
    
    # Test en escenarios específicos
    print("\n[4] Probando escenarios específicos:")
    
    # Escenario 1: Seguro - LiDAR despejado, bien alineado
    lidar_safe = np.full(180, 5.0)  # Todo despejado a 5m
    is_safe, conf = classifier.is_maneuver_safe(
        lidar_safe,
        robot_pos=np.array([2.0, 0.0]),
        robot_theta=0.0,
        box_pos=np.array([3.5, 0.1]),
        velocity=0.4
    )
    print(f"  Escenario SEGURO: is_safe={is_safe}, confidence={conf:.3f}")
    
    # Escenario 2: Peligroso - Obstáculo muy cerca
    lidar_danger = np.full(180, 0.3)  # Obstáculo a 30cm
    is_safe, conf = classifier.is_maneuver_safe(
        lidar_danger,
        robot_pos=np.array([2.0, 0.0]),
        robot_theta=0.0,
        box_pos=np.array([2.5, 0.0]),
        velocity=0.8
    )
    print(f"  Escenario PELIGROSO: is_safe={is_safe}, confidence={conf:.3f}")
    
    print("\n[OK] Test completado.")
