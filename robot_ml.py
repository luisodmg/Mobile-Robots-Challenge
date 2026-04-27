"""
robot_ml.py — Modelos ML simples para los 4 robots.

Integración minimalista con sim.py:
1. Husky - Logistic Regression (clasificar seguridad)
2. ANYmal - Linear Regression (predecir tiempo)
3. PuzzleBot - K-Means (zonas de trabajo)
4. Coordinator - Ridge Regression (tiempo total)
"""

import numpy as np


# ===========================================================================
# 1. HUSKY - Logistic Regression (Safety Classifier)
# ===========================================================================

class HuskyML:
    """Clasificador simple de seguridad para Husky."""
    
    def __init__(self):
        self.weights = np.array([0.5, -0.3, 0.2])  # [min_range, velocity, angle]
        self.bias = 0.1
        
    def is_safe(self, min_lidar_range: float, velocity: float, angle_to_box: float) -> tuple:
        """Retorna (is_safe, confidence)."""
        features = np.array([min_lidar_range, velocity, abs(angle_to_box)])
        z = np.dot(self.weights, features) + self.bias
        prob = 1 / (1 + np.exp(-z))
        return prob > 0.5, prob


# ===========================================================================
# 2. ANYMAL - Linear Regression (Time Predictor)
# ===========================================================================

class ANYmalML:
    """Predictor simple de tiempo para ANYmal."""
    
    def __init__(self):
        # Modelo: time = a*distance + b*payload + c
        self.weights = np.array([3.0, 0.5])  # [distance, payload]
        self.bias = 5.0
        
    def predict_time(self, distance: float, payload_kg: float) -> float:
        """Predice tiempo en segundos."""
        features = np.array([distance, payload_kg])
        time_pred = np.dot(self.weights, features) + self.bias
        return max(time_pred, 1.0)


# ===========================================================================
# 3. PUZZLEBOT - K-Means (Zone Discovery)
# ===========================================================================

class PuzzleBotML:
    """Descubridor simple de zonas para PuzzleBot."""
    
    def __init__(self):
        # Zonas predefinidas (centros)
        self.zones = {
            "pickup": np.array([9.8, 3.2]),
            "stack": np.array([10.5, 3.6]),
            "wait": np.array([9.0, 3.6])
        }
        
    def identify_zone(self, position: np.ndarray) -> str:
        """Identifica la zona más cercana."""
        min_dist = float('inf')
        closest_zone = "unknown"
        
        for zone_name, zone_center in self.zones.items():
            dist = np.linalg.norm(position[:2] - zone_center)
            if dist < min_dist:
                min_dist = dist
                closest_zone = zone_name
                
        return closest_zone


# ===========================================================================
# 4. COORDINATOR - Ridge Regression (Mission Time Predictor)
# ===========================================================================

class CoordinatorML:
    """Predictor simple de tiempo total de misión."""
    
    def __init__(self):
        # Modelo: total_time = w1*phase1 + w2*phase2 + w3*phase3 + bias
        self.weights = np.array([1.1, 1.05, 1.15])  # Factores por fase
        self.bias = 10.0  # Overhead
        
    def predict_total_time(self, phase1_time: float, phase2_time: float, phase3_time: float) -> float:
        """Predice tiempo total."""
        features = np.array([phase1_time, phase2_time, phase3_time])
        total = np.dot(self.weights, features) + self.bias
        return total


# ===========================================================================
# Wrapper Global para sim.py
# ===========================================================================

class RobotMLSystem:
    """Sistema ML completo para todos los robots."""
    
    def __init__(self):
        self.husky_ml = HuskyML()
        self.anymal_ml = ANYmalML()
        self.puzzlebot_ml = PuzzleBotML()
        self.coordinator_ml = CoordinatorML()
        
        print("[ML] Sistema ML inicializado para 4 robots")
        
    # Métodos de acceso directo
    
    def husky_check_safety(self, min_range: float, velocity: float, angle: float):
        """Husky: Verificar si maniobra es segura."""
        return self.husky_ml.is_safe(min_range, velocity, angle)
    
    def anymal_predict_eta(self, distance: float, payload: float):
        """ANYmal: Predecir tiempo de llegada."""
        return self.anymal_ml.predict_time(distance, payload)
    
    def puzzlebot_get_zone(self, position: np.ndarray):
        """PuzzleBot: Identificar zona actual."""
        return self.puzzlebot_ml.identify_zone(position)
    
    def coordinator_predict_mission(self, t1: float, t2: float, t3: float):
        """Coordinator: Predecir tiempo total de misión."""
        return self.coordinator_ml.predict_total_time(t1, t2, t3)


# Instancia global para sim.py
ml_system = RobotMLSystem()
