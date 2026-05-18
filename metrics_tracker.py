"""
metrics_tracker.py — Tracker de métricas para el hackathon IRS Inc. AI 2026.

Registra y reporta métricas clave:
- Makespan (tiempo total de operación)
- Número de replaneaciones
- Colisiones evitadas
- Eficiencia de flota
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import time


@dataclass
class CollisionEvent:
    """Evento de colisión evitada."""
    timestamp: float
    robot_id: int
    obstacle_distance: float
    avoidance_action: str  # "stop", "replan", "steer"


@dataclass
class ReplanEvent:
    """Evento de replaneación."""
    timestamp: float
    robot_id: int
    reason: str            # "obstacle_detected", "path_blocked", "dynamic_change"
    old_path_length: float
    new_path_length: float


@dataclass
class TaskEvent:
    """Evento de tarea."""
    timestamp: float
    task_id: int
    robot_id: int
    event_type: str        # "assigned", "started", "completed", "failed"


class MetricsTracker:
    """Tracker de métricas para el hackathon.
    
    Registra eventos y calcula métricas de desempeño del sistema multi-robot.
    """
    
    def __init__(self):
        # Tiempo
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Eventos
        self.collision_events: List[CollisionEvent] = []
        self.replan_events: List[ReplanEvent] = []
        self.task_events: List[TaskEvent] = []
        
        # Contadores
        self.total_collisions_avoided = 0
        self.total_replans = 0
        self.total_tasks_assigned = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # Métricas por robot
        self.robot_metrics: Dict[int, Dict] = {}
        
        # Distancias recorridas
        self.distances_traveled: Dict[int, float] = {}
        
    def start_mission(self):
        """Inicia el tracking de la misión."""
        self.start_time = time.time()
        print(f"[Metrics] Misión iniciada en t={self.start_time:.2f}")
    
    def end_mission(self):
        """Finaliza el tracking de la misión."""
        self.end_time = time.time()
        print(f"[Metrics] Misión finalizada en t={self.end_time:.2f}")
    
    def get_makespan(self) -> float:
        """Obtiene el makespan (tiempo total de operación).
        
        Returns:
            Tiempo total en segundos
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    # -----------------------------------------------------------------------
    # Registro de eventos
    # -----------------------------------------------------------------------
    
    def log_collision_avoided(self, robot_id: int, obstacle_distance: float, action: str = "stop"):
        """Registra una colisión evitada.
        
        Args:
            robot_id: ID del robot
            obstacle_distance: Distancia al obstáculo en metros
            action: Acción tomada para evitar colisión
        """
        event = CollisionEvent(
            timestamp=self._get_timestamp(),
            robot_id=robot_id,
            obstacle_distance=obstacle_distance,
            avoidance_action=action
        )
        self.collision_events.append(event)
        self.total_collisions_avoided += 1
        
        print(f"[Metrics] Robot {robot_id} evitó colisión (d={obstacle_distance:.2f}m, acción={action})")
    
    def log_replan(self, robot_id: int, reason: str, old_length: float = 0.0, new_length: float = 0.0):
        """Registra una replaneación.
        
        Args:
            robot_id: ID del robot
            reason: Razón de la replaneación
            old_length: Longitud del camino anterior
            new_length: Longitud del nuevo camino
        """
        event = ReplanEvent(
            timestamp=self._get_timestamp(),
            robot_id=robot_id,
            reason=reason,
            old_path_length=old_length,
            new_path_length=new_length
        )
        self.replan_events.append(event)
        self.total_replans += 1
        
        print(f"[Metrics] Robot {robot_id} replaneó ruta (razón={reason})")
    
    def log_task_event(self, task_id: int, robot_id: int, event_type: str):
        """Registra un evento de tarea.
        
        Args:
            task_id: ID de la tarea
            robot_id: ID del robot
            event_type: Tipo de evento ("assigned", "started", "completed", "failed")
        """
        event = TaskEvent(
            timestamp=self._get_timestamp(),
            task_id=task_id,
            robot_id=robot_id,
            event_type=event_type
        )
        self.task_events.append(event)
        
        if event_type == "assigned":
            self.total_tasks_assigned += 1
        elif event_type == "completed":
            self.total_tasks_completed += 1
        elif event_type == "failed":
            self.total_tasks_failed += 1
        
        print(f"[Metrics] Tarea {task_id} → {event_type} (robot {robot_id})")
    
    def log_distance_traveled(self, robot_id: int, distance: float):
        """Registra distancia recorrida por un robot.
        
        Args:
            robot_id: ID del robot
            distance: Distancia incremental en metros
        """
        if robot_id not in self.distances_traveled:
            self.distances_traveled[robot_id] = 0.0
        
        self.distances_traveled[robot_id] += distance
    
    def _get_timestamp(self) -> float:
        """Obtiene timestamp relativo al inicio de la misión."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    # -----------------------------------------------------------------------
    # Cálculo de métricas
    # -----------------------------------------------------------------------
    
    def calculate_fleet_efficiency(self) -> float:
        """Calcula la eficiencia de la flota.
        
        Eficiencia = tareas completadas / (tiempo total * número de robots)
        
        Returns:
            Eficiencia de flota
        """
        makespan = self.get_makespan()
        if makespan == 0:
            return 0.0
        
        n_robots = len(self.distances_traveled) if self.distances_traveled else 1
        
        efficiency = self.total_tasks_completed / (makespan * n_robots)
        return efficiency
    
    def calculate_avg_task_time(self) -> float:
        """Calcula el tiempo promedio por tarea.
        
        Returns:
            Tiempo promedio en segundos
        """
        if self.total_tasks_completed == 0:
            return 0.0
        
        # Calcular tiempo de cada tarea completada
        task_times = []
        
        for task_id in range(self.total_tasks_assigned):
            start_event = None
            end_event = None
            
            for event in self.task_events:
                if event.task_id == task_id:
                    if event.event_type in ["assigned", "started"]:
                        start_event = event
                    elif event.event_type == "completed":
                        end_event = event
            
            if start_event and end_event:
                task_time = end_event.timestamp - start_event.timestamp
                task_times.append(task_time)
        
        if not task_times:
            return 0.0
        
        return np.mean(task_times)
    
    def calculate_collision_rate(self) -> float:
        """Calcula la tasa de colisiones evitadas por minuto.
        
        Returns:
            Colisiones evitadas por minuto
        """
        makespan = self.get_makespan()
        if makespan == 0:
            return 0.0
        
        return (self.total_collisions_avoided / makespan) * 60.0
    
    def calculate_replan_rate(self) -> float:
        """Calcula la tasa de replaneaciones por minuto.
        
        Returns:
            Replaneaciones por minuto
        """
        makespan = self.get_makespan()
        if makespan == 0:
            return 0.0
        
        return (self.total_replans / makespan) * 60.0
    
    # -----------------------------------------------------------------------
    # Reportes
    # -----------------------------------------------------------------------
    
    def get_summary(self) -> Dict:
        """Obtiene un resumen de todas las métricas.
        
        Returns:
            Diccionario con métricas
        """
        return {
            # Tiempo
            "makespan_seconds": self.get_makespan(),
            "makespan_minutes": self.get_makespan() / 60.0,
            
            # Tareas
            "total_tasks_assigned": self.total_tasks_assigned,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "task_completion_rate": self.total_tasks_completed / max(1, self.total_tasks_assigned),
            "avg_task_time_seconds": self.calculate_avg_task_time(),
            
            # Navegación
            "total_collisions_avoided": self.total_collisions_avoided,
            "total_replans": self.total_replans,
            "collision_rate_per_min": self.calculate_collision_rate(),
            "replan_rate_per_min": self.calculate_replan_rate(),
            
            # Flota
            "fleet_efficiency": self.calculate_fleet_efficiency(),
            "total_distance_traveled": sum(self.distances_traveled.values()),
            "distances_by_robot": self.distances_traveled.copy(),
            
            # Eventos
            "collision_events": len(self.collision_events),
            "replan_events": len(self.replan_events),
            "task_events": len(self.task_events),
        }
    
    def print_report(self):
        """Imprime un reporte detallado de métricas."""
        summary = self.get_summary()
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*15 + "HACKATHON METRICS REPORT" + " "*29 + "║")
        print("╠" + "═"*68 + "╣")
        
        # Tiempo
        print("║  TIEMPO DE OPERACIÓN" + " "*47 + "║")
        print("║" + "-"*68 + "║")
        print(f"║  Makespan:                    {summary['makespan_seconds']:>8.2f} s ({summary['makespan_minutes']:>6.2f} min) ║")
        print("║" + " "*68 + "║")
        
        # Tareas
        print("║  TAREAS" + " "*60 + "║")
        print("║" + "-"*68 + "║")
        print(f"║  Asignadas:                   {summary['total_tasks_assigned']:>8} tareas" + " "*20 + "║")
        print(f"║  Completadas:                 {summary['total_tasks_completed']:>8} tareas" + " "*20 + "║")
        print(f"║  Fallidas:                    {summary['total_tasks_failed']:>8} tareas" + " "*20 + "║")
        print(f"║  Tasa de completitud:         {summary['task_completion_rate']:>8.1%}" + " "*27 + "║")
        print(f"║  Tiempo promedio por tarea:   {summary['avg_task_time_seconds']:>8.2f} s" + " "*23 + "║")
        print("║" + " "*68 + "║")
        
        # Navegación
        print("║  NAVEGACIÓN Y SEGURIDAD" + " "*44 + "║")
        print("║" + "-"*68 + "║")
        print(f"║  Colisiones evitadas:         {summary['total_collisions_avoided']:>8} eventos" + " "*19 + "║")
        print(f"║  Replaneaciones:              {summary['total_replans']:>8} eventos" + " "*19 + "║")
        print(f"║  Tasa de colisiones evitadas: {summary['collision_rate_per_min']:>8.2f} /min" + " "*20 + "║")
        print(f"║  Tasa de replaneaciones:      {summary['replan_rate_per_min']:>8.2f} /min" + " "*20 + "║")
        print("║" + " "*68 + "║")
        
        # Flota
        print("║  EFICIENCIA DE FLOTA" + " "*47 + "║")
        print("║" + "-"*68 + "║")
        print(f"║  Eficiencia:                  {summary['fleet_efficiency']:>8.4f}" + " "*27 + "║")
        print(f"║  Distancia total recorrida:   {summary['total_distance_traveled']:>8.2f} m" + " "*23 + "║")
        
        if summary['distances_by_robot']:
            print("║  Distancias por robot:" + " "*46 + "║")
            for robot_id, dist in summary['distances_by_robot'].items():
                print(f"║    Robot {robot_id}:                   {dist:>8.2f} m" + " "*27 + "║")
        
        print("╚" + "═"*68 + "╝")
    
    def export_to_dict(self) -> Dict:
        """Exporta todas las métricas y eventos a un diccionario.
        
        Returns:
            Diccionario completo con todos los datos
        """
        return {
            "summary": self.get_summary(),
            "collision_events": [
                {
                    "timestamp": e.timestamp,
                    "robot_id": e.robot_id,
                    "obstacle_distance": e.obstacle_distance,
                    "action": e.avoidance_action
                }
                for e in self.collision_events
            ],
            "replan_events": [
                {
                    "timestamp": e.timestamp,
                    "robot_id": e.robot_id,
                    "reason": e.reason,
                    "old_path_length": e.old_path_length,
                    "new_path_length": e.new_path_length
                }
                for e in self.replan_events
            ],
            "task_events": [
                {
                    "timestamp": e.timestamp,
                    "task_id": e.task_id,
                    "robot_id": e.robot_id,
                    "event_type": e.event_type
                }
                for e in self.task_events
            ]
        }


# Instancia global para el simulador
metrics_tracker = MetricsTracker()
