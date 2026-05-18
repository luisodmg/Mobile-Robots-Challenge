"""
task_allocator.py — Asignación de tareas para flota multi-robot.

Implementa estrategias de asignación:
1. Round-robin (simple)
2. Greedy por robot más cercano
3. Hungarian algorithm (opcional)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum, auto


class TaskStatus(Enum):
    """Estado de una tarea."""
    PENDING = auto()
    ASSIGNED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class Task:
    """Tarea de transporte."""
    task_id: int
    pickup_location: np.ndarray    # [x, y]
    dropoff_location: np.ndarray   # [x, y]
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[int] = None
    priority: int = 0              # Mayor = más prioritario
    
    def __repr__(self):
        return f"Task{self.task_id}({self.status.name}, robot={self.assigned_robot})"


@dataclass
class RobotState:
    """Estado de un robot para asignación."""
    robot_id: int
    position: np.ndarray           # [x, y]
    is_busy: bool = False
    current_task: Optional[int] = None
    
    def __repr__(self):
        return f"Robot{self.robot_id}@({self.position[0]:.2f},{self.position[1]:.2f})"


class TaskAllocator:
    """Asignador de tareas para flota multi-robot.
    
    Gestiona la asignación de tareas de transporte a robots disponibles.
    """
    
    def __init__(self, n_robots: int, strategy: str = "greedy"):
        """
        Args:
            n_robots: Número de robots en la flota
            strategy: Estrategia de asignación ("greedy", "round_robin", "hungarian")
        """
        self.n_robots = n_robots
        self.strategy = strategy
        
        # Tareas
        self.tasks: List[Task] = []
        self.task_counter = 0
        
        # Estados de robots
        self.robots: List[RobotState] = []
        for i in range(n_robots):
            self.robots.append(RobotState(robot_id=i, position=np.zeros(2)))
        
        # Estadísticas
        self.total_tasks_assigned = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.assignments_by_robot = [0] * n_robots
        
        # Round-robin counter
        self._rr_counter = 0
    
    def add_task(self, pickup: np.ndarray, dropoff: np.ndarray, priority: int = 0) -> int:
        """Agrega una nueva tarea a la cola.
        
        Args:
            pickup: Ubicación de recogida [x, y]
            dropoff: Ubicación de entrega [x, y]
            priority: Prioridad de la tarea (mayor = más prioritario)
            
        Returns:
            ID de la tarea creada
        """
        task = Task(
            task_id=self.task_counter,
            pickup_location=pickup.copy(),
            dropoff_location=dropoff.copy(),
            priority=priority
        )
        self.tasks.append(task)
        self.task_counter += 1
        return task.task_id
    
    def update_robot_position(self, robot_id: int, position: np.ndarray):
        """Actualiza la posición de un robot.
        
        Args:
            robot_id: ID del robot
            position: Nueva posición [x, y]
        """
        if 0 <= robot_id < self.n_robots:
            self.robots[robot_id].position = position.copy()
    
    def update_robot_status(self, robot_id: int, is_busy: bool, current_task: Optional[int] = None):
        """Actualiza el estado de ocupación de un robot.
        
        Args:
            robot_id: ID del robot
            is_busy: Si el robot está ocupado
            current_task: ID de la tarea actual (si aplica)
        """
        if 0 <= robot_id < self.n_robots:
            self.robots[robot_id].is_busy = is_busy
            self.robots[robot_id].current_task = current_task
    
    def assign_tasks(self) -> List[Tuple[int, int]]:
        """Asigna tareas pendientes a robots disponibles.
        
        Returns:
            Lista de tuplas (robot_id, task_id) con asignaciones nuevas
        """
        if self.strategy == "greedy":
            return self._assign_greedy()
        elif self.strategy == "round_robin":
            return self._assign_round_robin()
        elif self.strategy == "hungarian":
            return self._assign_hungarian()
        else:
            return self._assign_greedy()
    
    def _assign_greedy(self) -> List[Tuple[int, int]]:
        """Asignación greedy: robot más cercano toma la tarea más prioritaria.
        
        Returns:
            Lista de asignaciones (robot_id, task_id)
        """
        assignments = []
        
        # Obtener tareas pendientes ordenadas por prioridad
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Obtener robots disponibles
        available_robots = [r for r in self.robots if not r.is_busy]
        
        for task in pending_tasks:
            if not available_robots:
                break
            
            # Encontrar robot más cercano al pickup
            closest_robot = min(
                available_robots,
                key=lambda r: np.linalg.norm(r.position - task.pickup_location)
            )
            
            # Asignar tarea
            task.status = TaskStatus.ASSIGNED
            task.assigned_robot = closest_robot.robot_id
            closest_robot.is_busy = True
            closest_robot.current_task = task.task_id
            
            assignments.append((closest_robot.robot_id, task.task_id))
            available_robots.remove(closest_robot)
            
            self.total_tasks_assigned += 1
            self.assignments_by_robot[closest_robot.robot_id] += 1
        
        return assignments
    
    def _assign_round_robin(self) -> List[Tuple[int, int]]:
        """Asignación round-robin: tareas se asignan secuencialmente a robots.
        
        Returns:
            Lista de asignaciones (robot_id, task_id)
        """
        assignments = []
        
        # Obtener tareas pendientes
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        
        for task in pending_tasks:
            # Buscar siguiente robot disponible en round-robin
            attempts = 0
            while attempts < self.n_robots:
                robot = self.robots[self._rr_counter]
                self._rr_counter = (self._rr_counter + 1) % self.n_robots
                attempts += 1
                
                if not robot.is_busy:
                    # Asignar tarea
                    task.status = TaskStatus.ASSIGNED
                    task.assigned_robot = robot.robot_id
                    robot.is_busy = True
                    robot.current_task = task.task_id
                    
                    assignments.append((robot.robot_id, task.task_id))
                    
                    self.total_tasks_assigned += 1
                    self.assignments_by_robot[robot.robot_id] += 1
                    break
        
        return assignments
    
    def _assign_hungarian(self) -> List[Tuple[int, int]]:
        """Asignación Hungarian: minimiza costo total de asignación.
        
        Implementación simplificada sin scipy.
        
        Returns:
            Lista de asignaciones (robot_id, task_id)
        """
        # Para simplicidad, usar greedy como fallback
        # Una implementación completa requeriría scipy.optimize.linear_sum_assignment
        print("[TaskAllocator] Hungarian no implementado, usando greedy")
        return self._assign_greedy()
    
    def mark_task_in_progress(self, task_id: int):
        """Marca una tarea como en progreso."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.IN_PROGRESS
    
    def mark_task_completed(self, task_id: int):
        """Marca una tarea como completada y libera el robot."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            self.total_tasks_completed += 1
            
            # Liberar robot
            if task.assigned_robot is not None:
                self.update_robot_status(task.assigned_robot, is_busy=False, current_task=None)
    
    def mark_task_failed(self, task_id: int):
        """Marca una tarea como fallida y libera el robot."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            self.total_tasks_failed += 1
            
            # Liberar robot
            if task.assigned_robot is not None:
                self.update_robot_status(task.assigned_robot, is_busy=False, current_task=None)
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Obtiene una tarea por ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_pending_tasks(self) -> List[Task]:
        """Obtiene todas las tareas pendientes."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]
    
    def get_active_tasks(self) -> List[Task]:
        """Obtiene tareas asignadas o en progreso."""
        return [t for t in self.tasks if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]]
    
    def get_completed_tasks(self) -> List[Task]:
        """Obtiene tareas completadas."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
    
    def get_robot_task(self, robot_id: int) -> Optional[Task]:
        """Obtiene la tarea actual de un robot."""
        if 0 <= robot_id < self.n_robots:
            task_id = self.robots[robot_id].current_task
            if task_id is not None:
                return self.get_task(task_id)
        return None
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas del asignador."""
        return {
            "strategy": self.strategy,
            "n_robots": self.n_robots,
            "total_tasks": len(self.tasks),
            "pending_tasks": len(self.get_pending_tasks()),
            "active_tasks": len(self.get_active_tasks()),
            "completed_tasks": self.total_tasks_completed,
            "failed_tasks": self.total_tasks_failed,
            "total_assigned": self.total_tasks_assigned,
            "assignments_by_robot": self.assignments_by_robot.copy(),
            "busy_robots": sum(1 for r in self.robots if r.is_busy)
        }
    
    def print_status(self):
        """Imprime el estado actual del asignador."""
        print("\n" + "="*60)
        print(f"  TASK ALLOCATOR STATUS ({self.strategy.upper()})")
        print("="*60)
        
        stats = self.get_statistics()
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"  Pending:   {stats['pending_tasks']}")
        print(f"  Active:    {stats['active_tasks']}")
        print(f"  Completed: {stats['completed_tasks']}")
        print(f"  Failed:    {stats['failed_tasks']}")
        print(f"\nRobots busy: {stats['busy_robots']}/{self.n_robots}")
        print(f"Assignments by robot: {stats['assignments_by_robot']}")
        print("="*60)


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def create_default_tasks() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Crea un conjunto de tareas por defecto para el almacén.
    
    Returns:
        Lista de tuplas (pickup, dropoff)
    """
    tasks = [
        # Tareas de apilado de cajas pequeñas
        (np.array([9.5, 3.2]), np.array([10.5, 3.6])),  # Caja A
        (np.array([9.8, 3.2]), np.array([10.5, 3.6])),  # Caja B
        (np.array([10.1, 3.2]), np.array([10.5, 3.6])), # Caja C
    ]
    return tasks
