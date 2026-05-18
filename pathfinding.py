"""
pathfinding.py — Planificación de rutas con A* para navegación multi-robot.

Implementa A* sobre un grid 2D del mapa conocido.
Soporta replaneación dinámica ante obstáculos detectados visualmente.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq


@dataclass
class GridNode:
    """Nodo en el grid de planificación."""
    x: int              # Coordenada x en grid
    y: int              # Coordenada y en grid
    g: float = np.inf   # Costo desde inicio
    h: float = 0.0      # Heurística a meta
    parent: Optional['GridNode'] = None
    
    @property
    def f(self) -> float:
        """Costo total f = g + h"""
        return self.g + self.h
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class AStarPlanner:
    """Planificador A* para navegación en grid 2D.
    
    Características:
    - Grid discretizado del mapa continuo
    - Heurística euclidiana
    - Movimientos en 8 direcciones
    - Replaneación dinámica
    """
    
    def __init__(self, map_bounds: Tuple[float, float, float, float], resolution: float = 0.1):
        """
        Args:
            map_bounds: (x_min, x_max, y_min, y_max) en metros
            resolution: Tamaño de celda del grid en metros
        """
        self.x_min, self.x_max, self.y_min, self.y_max = map_bounds
        self.resolution = resolution
        
        # Dimensiones del grid
        self.grid_width = int((self.x_max - self.x_min) / resolution)
        self.grid_height = int((self.y_max - self.y_min) / resolution)
        
        # Grid de ocupación (True = ocupado, False = libre)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Obstáculos estáticos (del mapa conocido)
        self.static_obstacles: List[Tuple[int, int]] = []
        
        # Estadísticas
        self.total_plans = 0
        self.total_replans = 0
        self.total_nodes_expanded = 0
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convierte coordenadas del mundo a coordenadas del grid.
        
        Args:
            x, y: Coordenadas en metros
            
        Returns:
            (gx, gy) coordenadas en grid
        """
        gx = int((x - self.x_min) / self.resolution)
        gy = int((y - self.y_min) / self.resolution)
        
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        
        return gx, gy
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convierte coordenadas del grid a coordenadas del mundo.
        
        Args:
            gx, gy: Coordenadas en grid
            
        Returns:
            (x, y) coordenadas en metros
        """
        x = self.x_min + (gx + 0.5) * self.resolution
        y = self.y_min + (gy + 0.5) * self.resolution
        return x, y
    
    def add_static_obstacle(self, x: float, y: float, radius: float = 0.3):
        """Agrega un obstáculo estático al grid.
        
        Args:
            x, y: Posición del obstáculo en metros
            radius: Radio del obstáculo en metros
        """
        gx, gy = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution)
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if dx*dx + dy*dy <= grid_radius*grid_radius:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.grid[ny, nx] = True
                        self.static_obstacles.append((nx, ny))
    
    def add_dynamic_obstacle(self, x: float, y: float, radius: float = 0.2):
        """Agrega un obstáculo dinámico (detectado visualmente) al grid.
        
        Args:
            x, y: Posición del obstáculo en metros
            radius: Radio del obstáculo en metros
        """
        self.add_static_obstacle(x, y, radius)
    
    def clear_dynamic_obstacles(self):
        """Limpia obstáculos dinámicos, manteniendo solo los estáticos."""
        self.grid.fill(False)
        for gx, gy in self.static_obstacles:
            if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                self.grid[gy, gx] = True
    
    def is_valid(self, gx: int, gy: int) -> bool:
        """Verifica si una celda del grid es válida y libre.
        
        Args:
            gx, gy: Coordenadas en grid
            
        Returns:
            True si la celda es válida y libre
        """
        if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
            return False
        return not self.grid[gy, gx]
    
    def get_neighbors(self, node: GridNode) -> List[GridNode]:
        """Obtiene vecinos válidos de un nodo (8 direcciones).
        
        Args:
            node: Nodo actual
            
        Returns:
            Lista de nodos vecinos válidos
        """
        neighbors = []
        
        # 8 direcciones: N, NE, E, SE, S, SW, W, NW
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            
            if self.is_valid(nx, ny):
                # Costo de movimiento (diagonal = √2, recto = 1)
                cost = np.sqrt(dx*dx + dy*dy)
                neighbor = GridNode(nx, ny)
                neighbors.append((neighbor, cost))
        
        return neighbors
    
    def heuristic(self, node: GridNode, goal: GridNode) -> float:
        """Calcula la heurística (distancia euclidiana).
        
        Args:
            node: Nodo actual
            goal: Nodo meta
            
        Returns:
            Distancia heurística
        """
        dx = node.x - goal.x
        dy = node.y - goal.y
        return np.sqrt(dx*dx + dy*dy)
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Planifica una ruta desde start hasta goal usando A*.
        
        Args:
            start: Posición inicial (x, y) en metros
            goal: Posición meta (x, y) en metros
            
        Returns:
            Lista de waypoints (x, y) en metros, o None si no hay ruta
        """
        self.total_plans += 1
        
        # Convertir a coordenadas de grid
        start_gx, start_gy = self.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.world_to_grid(goal[0], goal[1])
        
        # Verificar que inicio y meta sean válidos
        if not self.is_valid(start_gx, start_gy):
            print(f"[A*] Inicio ({start[0]:.2f}, {start[1]:.2f}) no es válido")
            return None
        
        if not self.is_valid(goal_gx, goal_gy):
            print(f"[A*] Meta ({goal[0]:.2f}, {goal[1]:.2f}) no es válida")
            return None
        
        # Inicializar nodos
        start_node = GridNode(start_gx, start_gy, g=0.0)
        goal_node = GridNode(goal_gx, goal_gy)
        start_node.h = self.heuristic(start_node, goal_node)
        
        # Open set (heap) y closed set
        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        
        # Diccionario de nodos visitados
        nodes_dict = {(start_gx, start_gy): start_node}
        
        nodes_expanded = 0
        
        while open_set:
            # Obtener nodo con menor f
            current = heapq.heappop(open_set)
            nodes_expanded += 1
            
            # Verificar si llegamos a la meta
            if current.x == goal_node.x and current.y == goal_node.y:
                self.total_nodes_expanded += nodes_expanded
                return self._reconstruct_path(current)
            
            # Marcar como visitado
            closed_set.add((current.x, current.y))
            
            # Explorar vecinos
            for neighbor, move_cost in self.get_neighbors(current):
                if (neighbor.x, neighbor.y) in closed_set:
                    continue
                
                tentative_g = current.g + move_cost
                
                # Obtener o crear nodo vecino
                neighbor_key = (neighbor.x, neighbor.y)
                if neighbor_key not in nodes_dict:
                    neighbor.h = self.heuristic(neighbor, goal_node)
                    nodes_dict[neighbor_key] = neighbor
                else:
                    neighbor = nodes_dict[neighbor_key]
                
                # Actualizar si encontramos un mejor camino
                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.parent = current
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        # No se encontró ruta
        self.total_nodes_expanded += nodes_expanded
        print(f"[A*] No se encontró ruta de ({start[0]:.2f}, {start[1]:.2f}) a ({goal[0]:.2f}, {goal[1]:.2f})")
        return None
    
    def _reconstruct_path(self, goal_node: GridNode) -> List[Tuple[float, float]]:
        """Reconstruye el camino desde el nodo meta hasta el inicio.
        
        Args:
            goal_node: Nodo meta alcanzado
            
        Returns:
            Lista de waypoints en coordenadas del mundo
        """
        path = []
        current = goal_node
        
        while current is not None:
            x, y = self.grid_to_world(current.x, current.y)
            path.append((x, y))
            current = current.parent
        
        path.reverse()
        return path
    
    def simplify_path(self, path: List[Tuple[float, float]], epsilon: float = 0.3) -> List[Tuple[float, float]]:
        """Simplifica un camino eliminando waypoints redundantes.
        
        Args:
            path: Camino original
            epsilon: Tolerancia para simplificación
            
        Returns:
            Camino simplificado
        """
        if len(path) <= 2:
            return path
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = np.array(simplified[-1])
            curr = np.array(path[i])
            next_pt = np.array(path[i + 1])
            
            # Calcular distancia perpendicular
            line_vec = next_pt - prev
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                continue
            
            line_unitvec = line_vec / line_len
            point_vec = curr - prev
            proj_length = np.dot(point_vec, line_unitvec)
            proj = prev + proj_length * line_unitvec
            dist = np.linalg.norm(curr - proj)
            
            if dist > epsilon:
                simplified.append(tuple(curr))
        
        simplified.append(path[-1])
        return simplified
    
    def replan(self, current_pos: Tuple[float, float], goal: Tuple[float, float],
               new_obstacles: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """Replanifica una ruta considerando nuevos obstáculos detectados.
        
        Args:
            current_pos: Posición actual
            goal: Meta
            new_obstacles: Lista de nuevos obstáculos detectados (x, y)
            
        Returns:
            Nueva ruta o None
        """
        self.total_replans += 1
        
        # Agregar nuevos obstáculos al grid
        for obs_x, obs_y in new_obstacles:
            self.add_dynamic_obstacle(obs_x, obs_y)
        
        # Planificar nueva ruta
        new_path = self.plan(current_pos, goal)
        
        return new_path
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas del planificador."""
        return {
            "total_plans": self.total_plans,
            "total_replans": self.total_replans,
            "total_nodes_expanded": self.total_nodes_expanded,
            "avg_nodes_per_plan": self.total_nodes_expanded / max(1, self.total_plans),
            "grid_size": (self.grid_width, self.grid_height),
            "resolution": self.resolution,
            "static_obstacles": len(self.static_obstacles)
        }


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def create_warehouse_planner(resolution: float = 0.1) -> AStarPlanner:
    """Crea un planificador A* para el mapa del almacén.
    
    Args:
        resolution: Resolución del grid en metros
        
    Returns:
        AStarPlanner configurado
    """
    # Bounds del mapa del almacén
    map_bounds = (-1.0, 13.0, -3.0, 6.0)
    
    planner = AStarPlanner(map_bounds, resolution)
    
    # Agregar obstáculos estáticos conocidos (paredes, zonas prohibidas)
    # Estos se pueden agregar según el mapa conocido
    
    return planner
