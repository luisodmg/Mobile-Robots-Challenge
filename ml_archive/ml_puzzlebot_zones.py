"""
ml_puzzlebot_zones.py — Integración de K-Means para PuzzleBot.

Descubrimiento de zonas operacionales sin etiquetas:
- Input: historial de posiciones, configuraciones del brazo, frecuencia de tareas
- Output: cluster labels, centros de zonas (pickup, stacking, waiting, navigation)
"""

import numpy as np
from typing import List, Tuple, Dict
from ml_models import KMeans


class PuzzleBotZoneDiscovery:
    """Descubridor ML de zonas operacionales en el workspace del PuzzleBot."""
    
    def __init__(self, n_zones: int = 4):
        """
        Args:
            n_zones: número de zonas a descubrir (default=4: pickup, stack, wait, nav)
        """
        self.model = KMeans(n_clusters=n_zones, max_iter=100)
        self.n_zones = n_zones
        self.zone_names = {}  # Mapeo de cluster_id -> nombre interpretado
        self.trained = False
        
    def extract_features(
        self,
        position: np.ndarray,
        arm_extension: float,
        task_frequency: float,
        time_spent: float
    ) -> np.ndarray:
        """Extrae features de una observación del PuzzleBot.
        
        Features:
        1. x_position: coordenada X en el workspace
        2. y_position: coordenada Y en el workspace
        3. arm_extension: extensión del brazo [0-1 normalizado]
        4. task_frequency: frecuencia de tareas en esta posición [0-1]
        5. time_spent: tiempo promedio pasado aquí [normalizado]
        """
        features = np.array([
            position[0],
            position[1],
            arm_extension,
            task_frequency,
            time_spent
        ])
        return features
    
    def generate_workspace_data(
        self,
        n_samples: int = 500,
        workspace_bounds: Tuple[float, float, float, float] = (8.5, 11.0, 3.0, 4.2)
    ) -> np.ndarray:
        """Genera datos sintéticos de actividad del PuzzleBot en el workspace.
        
        Simula 4 zonas naturales:
        1. Pickup zone: cerca de la mesa de cajas (x~9.8, y~3.2)
        2. Stacking zone: área de apilado (x~10.5, y~3.6)
        3. Waiting zone: área de espera/deploy (x~9.0, y~3.6)
        4. Navigation corridors: rutas entre zonas
        """
        X = []
        x_min, x_max, y_min, y_max = workspace_bounds
        
        # Zona 1: Pickup (alta frecuencia de tareas, brazo extendido)
        n_pickup = n_samples // 4
        for _ in range(n_pickup):
            x = np.random.normal(9.8, 0.15)  # Centrado en mesa de cajas
            y = np.random.normal(3.2, 0.1)
            arm_ext = np.random.uniform(0.6, 1.0)  # Brazo extendido
            task_freq = np.random.uniform(0.7, 1.0)  # Alta frecuencia
            time_spent = np.random.uniform(0.3, 0.6)  # Tiempo medio
            
            X.append([x, y, arm_ext, task_freq, time_spent])
        
        # Zona 2: Stacking (muy alta frecuencia, brazo muy extendido)
        n_stack = n_samples // 4
        for _ in range(n_stack):
            x = np.random.normal(10.5, 0.1)  # Área de apilado
            y = np.random.normal(3.6, 0.1)
            arm_ext = np.random.uniform(0.7, 1.0)  # Brazo muy extendido
            task_freq = np.random.uniform(0.8, 1.0)  # Muy alta frecuencia
            time_spent = np.random.uniform(0.4, 0.8)  # Tiempo alto
            
            X.append([x, y, arm_ext, task_freq, time_spent])
        
        # Zona 3: Waiting/Deploy (baja frecuencia, brazo retraído)
        n_wait = n_samples // 4
        for _ in range(n_wait):
            x = np.random.normal(9.0, 0.15)  # Área de espera
            y = np.random.uniform(3.2, 4.0)
            arm_ext = np.random.uniform(0.0, 0.3)  # Brazo retraído
            task_freq = np.random.uniform(0.0, 0.3)  # Baja frecuencia
            time_spent = np.random.uniform(0.1, 0.4)  # Tiempo bajo
            
            X.append([x, y, arm_ext, task_freq, time_spent])
        
        # Zona 4: Navigation corridors (frecuencia media, brazo retraído, poco tiempo)
        n_nav = n_samples - n_pickup - n_stack - n_wait
        for _ in range(n_nav):
            x = np.random.uniform(x_min, x_max)  # Distribuido por todo el workspace
            y = np.random.uniform(y_min, y_max)
            arm_ext = np.random.uniform(0.0, 0.2)  # Brazo retraído
            task_freq = np.random.uniform(0.1, 0.5)  # Frecuencia media-baja
            time_spent = np.random.uniform(0.0, 0.2)  # Poco tiempo (pasando)
            
            X.append([x, y, arm_ext, task_freq, time_spent])
        
        return np.array(X)
    
    def train(self, X: np.ndarray):
        """Entrena K-Means para descubrir zonas en el workspace."""
        print(f"\n[PuzzleBotZoneDiscovery] Entrenando K-Means con {self.n_zones} clusters...")
        
        # Entrenar
        labels = self.model.fit(X)
        
        print(f"  Inertia: {self.model.inertia:.4f}")
        print(f"  Centros de clusters:")
        for i, center in enumerate(self.model.centers):
            print(f"    Cluster {i}: pos=({center[0]:.2f}, {center[1]:.2f}), "
                  f"arm_ext={center[2]:.2f}, task_freq={center[3]:.2f}, time={center[4]:.2f}")
        
        # Interpretar clusters automáticamente
        self._interpret_clusters(X, labels)
        
        self.trained = True
        return labels
    
    def _interpret_clusters(self, X: np.ndarray, labels: np.ndarray):
        """Interpreta automáticamente qué representa cada cluster."""
        for cluster_id in range(self.n_zones):
            cluster_points = X[labels == cluster_id]
            
            if len(cluster_points) == 0:
                self.zone_names[cluster_id] = "empty"
                continue
            
            # Promedios del cluster
            avg_x = np.mean(cluster_points[:, 0])
            avg_y = np.mean(cluster_points[:, 1])
            avg_arm_ext = np.mean(cluster_points[:, 2])
            avg_task_freq = np.mean(cluster_points[:, 3])
            avg_time = np.mean(cluster_points[:, 4])
            
            # Heurísticas de interpretación
            if avg_task_freq > 0.7 and avg_arm_ext > 0.6:
                if avg_x > 10.0:
                    name = "stacking_zone"
                else:
                    name = "pickup_zone"
            elif avg_task_freq < 0.4 and avg_arm_ext < 0.3:
                if avg_time < 0.3:
                    name = "navigation_corridor"
                else:
                    name = "waiting_zone"
            else:
                name = f"zone_{cluster_id}"
            
            self.zone_names[cluster_id] = name
            
        print("\n  Interpretación de zonas:")
        for cluster_id, name in self.zone_names.items():
            print(f"    Cluster {cluster_id} → {name}")
    
    def predict_zone(
        self,
        position: np.ndarray,
        arm_extension: float,
        task_frequency: float,
        time_spent: float
    ) -> Tuple[int, str]:
        """Predice a qué zona pertenece una nueva observación.
        
        Returns:
            (cluster_id, zone_name)
        """
        if not self.trained:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        features = self.extract_features(position, arm_extension, task_frequency, time_spent)
        features = features.reshape(1, -1)
        
        cluster_id = self.model.predict(features)[0]
        zone_name = self.zone_names.get(cluster_id, f"zone_{cluster_id}")
        
        return int(cluster_id), zone_name
    
    def get_zone_centers(self) -> Dict[str, np.ndarray]:
        """Retorna los centros de cada zona descubierta."""
        if not self.trained:
            return {}
        
        centers = {}
        for cluster_id, name in self.zone_names.items():
            centers[name] = self.model.centers[cluster_id]
        
        return centers
    
    def get_safe_navigation_path(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray
    ) -> List[np.ndarray]:
        """Sugiere waypoints seguros basados en zonas descubiertas.
        
        Evita zonas de alta actividad (pickup/stacking) y prefiere corredores.
        """
        if not self.trained:
            return [start_pos, goal_pos]
        
        # Identificar zonas de navegación segura
        safe_zones = []
        for cluster_id, name in self.zone_names.items():
            if "navigation" in name or "waiting" in name:
                safe_zones.append(self.model.centers[cluster_id][:2])  # Solo x,y
        
        if not safe_zones:
            return [start_pos, goal_pos]
        
        # Encontrar el waypoint más cercano a la línea recta
        mid_point = (start_pos[:2] + goal_pos[:2]) / 2
        distances = [np.linalg.norm(zone - mid_point) for zone in safe_zones]
        best_waypoint_idx = np.argmin(distances)
        waypoint = safe_zones[best_waypoint_idx]
        
        # Path: start → waypoint → goal
        path = [
            start_pos,
            np.array([waypoint[0], waypoint[1], 0.0]),
            goal_pos
        ]
        
        return path
    
    def visualize_zones(self) -> str:
        """Retorna una representación ASCII del workspace con zonas."""
        if not self.trained:
            return "Modelo no entrenado."
        
        # Grid ASCII simple
        grid = [[' ' for _ in range(40)] for _ in range(20)]
        
        # Mapear coordenadas del mundo a grid
        # Workspace: x=[8.5, 11.0], y=[3.0, 4.2]
        def world_to_grid(x, y):
            grid_x = int((x - 8.5) / 2.5 * 39)
            grid_y = int((y - 3.0) / 1.2 * 19)
            grid_x = np.clip(grid_x, 0, 39)
            grid_y = np.clip(grid_y, 0, 19)
            return grid_x, 19 - grid_y  # Invertir Y para visualización
        
        # Marcar centros de zonas
        symbols = ['P', 'S', 'W', 'N']  # Pickup, Stack, Wait, Nav
        for cluster_id, name in self.zone_names.items():
            center = self.model.centers[cluster_id]
            gx, gy = world_to_grid(center[0], center[1])
            
            if "pickup" in name:
                symbol = 'P'
            elif "stack" in name:
                symbol = 'S'
            elif "wait" in name:
                symbol = 'W'
            elif "nav" in name:
                symbol = 'N'
            else:
                symbol = str(cluster_id)
            
            grid[gy][gx] = symbol
        
        # Convertir a string
        result = "  Workspace Zones (P=Pickup, S=Stack, W=Wait, N=Nav):\n"
        result += "  +" + "-" * 40 + "+\n"
        for row in grid:
            result += "  |" + "".join(row) + "|\n"
        result += "  +" + "-" * 40 + "+\n"
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Test: PuzzleBot Zone Discovery (K-Means Clustering)")
    print("=" * 60)
    
    np.random.seed(42)
    
    zone_discovery = PuzzleBotZoneDiscovery(n_zones=4)
    
    # Generar datos de workspace
    print("\n[1] Generando datos de actividad del workspace...")
    X = zone_discovery.generate_workspace_data(n_samples=600)
    print(f"  Generadas {len(X)} observaciones")
    print(f"  Features: [x, y, arm_extension, task_frequency, time_spent]")
    
    # Entrenar
    print("\n[2] Descubriendo zonas con K-Means...")
    labels = zone_discovery.train(X)
    
    # Distribución de clusters
    print("\n[3] Distribución de puntos por zona:")
    for cluster_id in range(zone_discovery.n_zones):
        count = np.sum(labels == cluster_id)
        name = zone_discovery.zone_names[cluster_id]
        print(f"  {name:25s}: {count:3d} puntos ({100*count/len(X):.1f}%)")
    
    # Centros de zonas
    print("\n[4] Centros de zonas descubiertas:")
    centers = zone_discovery.get_zone_centers()
    for name, center in centers.items():
        print(f"  {name:25s}: pos=({center[0]:.2f}, {center[1]:.2f})")
    
    # Test de predicción
    print("\n[5] Probando predicción de zona:")
    
    # Punto en zona de pickup
    cluster_id, zone_name = zone_discovery.predict_zone(
        position=np.array([9.8, 3.2, 0.0]),
        arm_extension=0.8,
        task_frequency=0.9,
        time_spent=0.5
    )
    print(f"  Pos (9.8, 3.2) con brazo extendido → {zone_name}")
    
    # Punto en zona de navegación
    cluster_id, zone_name = zone_discovery.predict_zone(
        position=np.array([9.5, 3.8, 0.0]),
        arm_extension=0.1,
        task_frequency=0.2,
        time_spent=0.1
    )
    print(f"  Pos (9.5, 3.8) pasando rápido → {zone_name}")
    
    # Path planning
    print("\n[6] Generando path seguro:")
    start = np.array([9.0, 3.6, 0.0])
    goal = np.array([10.5, 3.6, 0.0])
    path = zone_discovery.get_safe_navigation_path(start, goal)
    print(f"  Start: ({start[0]:.2f}, {start[1]:.2f})")
    for i, waypoint in enumerate(path[1:-1], 1):
        print(f"  Waypoint {i}: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
    print(f"  Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
    
    # Visualización
    print("\n[7] Visualización del workspace:")
    print(zone_discovery.visualize_zones())
    
    print("[OK] Test completado.")
