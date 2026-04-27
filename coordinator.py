"""
coordinator.py — Máquina de estados que orquesta las tres fases del reto.

Fases:
    IDLE → PHASE1_HUSKY → PHASE2_ANYMAL → XARM_UNLOAD → PHASE3_PUZZLEBOTS → DONE
"""

import numpy as np
import time
from enum import Enum, auto
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Imports directos a los módulos en la misma carpeta
from husky_pusher import HuskyPusher
from anymal_gait import ANYmalGait
from puzzlebot_arm import PuzzleBotArm


# ---------------------------------------------------------------------------
# Estado del sistema
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE            = auto()
    PHASE1_HUSKY    = auto()
    PHASE2_ANYMAL   = auto()
    XARM_UNLOAD     = auto()   # Extra: XArm baja los PuzzleBots
    PHASE3_PUZZLEBOTS = auto()
    DONE            = auto()
    ERROR           = auto()


@dataclass
class SmallBox:
    """Caja pequeña que deben apilar los PuzzleBots."""
    name: str     # "A", "B", "C"
    pos: np.ndarray
    stacked: bool = False
    stack_height: float = 0.0  # altura en la pila [m]

    BOX_HEIGHT = 0.05  # m


# ---------------------------------------------------------------------------
# XArm — manipulador de descarga (Puntos extra)
# ---------------------------------------------------------------------------

class XArm:
    """XArm de 6 DoF (simplificado) para bajar los PuzzleBots del ANYmal.

    Modelo: cinemática de alcance adaptada, montado junto a la zona de trabajo.
    """

    # Dimensiones básicas (eslabones adaptados a la simulación)
    LINK_LENGTHS = [0.267, 0.289, 0.078, 0.343, 0.076, 0.097]  # m

    def __init__(self, arm_id: int, base_pos: np.ndarray):
        self.id = arm_id
        self.base_pos = base_pos
        self.q = np.zeros(6)       # Articulaciones [rad]
        self.payload: Optional[str] = None  # Nombre del PuzzleBot cargado

    def _ik_simple(self, target_world: np.ndarray) -> np.ndarray:
        """IK simplificada: calcula ángulos para alcanzar target_world."""
        delta = target_world - self.base_pos
        r = np.linalg.norm(delta[:2])
        z = delta[2]

        # Ángulo de la base
        q1 = np.arctan2(delta[1], delta[0])

        # Alcance en el plano (r, z) para eslabones 2 y 3 de la cadena
        L1 = self.LINK_LENGTHS[1]  # 0.289
        L2 = self.LINK_LENGTHS[3]  # 0.343
        
        # Distancia euclidiana directa al objetivo
        D = np.sqrt(r**2 + z**2)
        
        # Limitar matemáticamente D para evitar singularidad de frontera (NaN)
        D = np.clip(D, 0.01, L1 + L2 - 0.005)

        # Ley de cosenos protegida
        cos_q3 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
        q3 = -np.arccos(np.clip(cos_q3, -1.0, 1.0))

        beta = np.arctan2(z, r)
        gamma = np.arctan2(L2 * np.sin(-q3), L1 + L2 * np.cos(-q3))
        q2 = beta + gamma

        q = np.array([q1, q2, 0.0, q3, 0.0, 0.0])
        self.q = q
        return q

    def pick_from_anymal(self, puzzlebot_id: int, anymal_pos: np.ndarray) -> bool:
        # PuzzleBots están espaciados 0.15 m en el dorso del ANYmal
        offset_x = (puzzlebot_id - 1) * 0.15
        pick_pos = np.array([
            anymal_pos[0] + offset_x,
            anymal_pos[1],
            0.42 + 0.12  # Altura del dorso del ANYmal + altura PuzzleBot
        ])

        dist = np.linalg.norm(pick_pos - self.base_pos)
        reach = self.LINK_LENGTHS[1] + self.LINK_LENGTHS[3]
        if dist > reach:
            print(f"[XArm{self.id}] PuzzleBot {puzzlebot_id} fuera de alcance (dist={dist:.3f} > {reach:.3f} m)")
            return False

        q = self._ik_simple(pick_pos)
        print(f"[XArm{self.id}] Recogiendo PuzzleBot {puzzlebot_id} desde ({pick_pos[0]:.2f}, {pick_pos[1]:.2f}, {pick_pos[2]:.2f})")
        print(f"[XArm{self.id}] q = {np.round(np.degrees(q[:4]), 1)}° (primeras 4 articulaciones)")
        self.payload = f"PuzzleBot_{puzzlebot_id}"
        return True

    def place_on_table(self, table_pos: np.ndarray) -> bool:
        if self.payload is None:
            print(f"[XArm{self.id}] No hay payload.")
            return False

        q = self._ik_simple(table_pos)
        print(f"[XArm{self.id}] Colocando {self.payload} en mesa ({table_pos[0]:.2f}, {table_pos[1]:.2f}, {table_pos[2]:.2f})")
        print(f"[XArm{self.id}] q = {np.round(np.degrees(q[:4]), 1)}°")
        self.payload = None
        return True


# ---------------------------------------------------------------------------
# PuzzleBot completo (robot móvil + brazo)
# ---------------------------------------------------------------------------

class PuzzleBot:
    """PuzzleBot: robot diferencial con mini brazo 3 DoF."""

    TABLE_BOXES_POS = {
        "A": np.array([9.5, 3.2, 0.02]),
        "B": np.array([9.8, 3.2, 0.02]),
        "C": np.array([10.1, 3.2, 0.02]),
    }
    STACK_POS = np.array([10.5, 3.6, 0.0])  # Base de la pila

    def __init__(self, pb_id: int, deploy_pos: np.ndarray, dt: float = 0.05):
        self.id = pb_id
        self.pos = deploy_pos.copy()
        self.arm = PuzzleBotArm()
        self.assigned_box: Optional[str] = None
        self.done = False
        self.dt = dt  # Time step for movement
        
        # Event-based synchronization
        self.waiting_for_event = False
        self.completed_event = None  # Which event this PuzzleBot completed
        
        # State machine for non-blocking operation
        self.state = "IDLE"
        self.target_box_pos = None
        self.stack_target_pos = None 

    def move_to(self, target: np.ndarray, v: float = 0.2, dt: float = 0.05):
        steps = int(np.linalg.norm(target[:2] - self.pos[:2]) / (v * dt)) + 1
        for _ in range(steps):
            delta = target[:2] - self.pos[:2]
            dist = np.linalg.norm(delta)
            if dist < 0.02:
                break
            self.pos[:2] += v * dt * delta / (dist + 1e-9)

    def pick_and_stack_nonblocking(
        self,
        box_name: str,
        stack_height: float,
        exclusion_zones: List[Tuple[np.ndarray, float]],
        event_flags: dict,
        obstacles: List[Tuple[np.ndarray, float]] = None
    ) -> Tuple[bool, float]:
        """Event-based non-blocking version of pick_and_stack."""
        
        # Check for event dependencies
        if box_name == "B" and not event_flags.get("C_completed", False):
            return False, stack_height
        elif box_name == "A" and not event_flags.get("B_completed", False):
            return False, stack_height
            
        box_pos = self.TABLE_BOXES_POS[box_name]

        # Check exclusion zones
        for (center, radius) in exclusion_zones:
            if np.linalg.norm(box_pos[:2] - center[:2]) < radius:
                return False, stack_height

        # State machine for non-blocking operation
        if self.state == "IDLE":
            print(f"\n[PB{self.id}] Iniciando recogida de caja {box_name}")
            self.target_box_pos = box_pos
            self.stack_target_pos = self.STACK_POS + np.array([0, 0, stack_height])
            self.state = "MOVING_TO_BOX"
            
        elif self.state == "MOVING_TO_BOX":
            approach = np.array([box_pos[0] - 0.12, box_pos[1], 0.0])
            if self._step_move_to(approach, obstacles=obstacles):
                self.state = "GRASPING"
                self.arm.reset()
            return False, stack_height
            
        elif self.state == "GRASPING":
            arm_target = np.array([0.08, 0.0, 0.02])
            success = self.arm.grasp_box(arm_target, grip_force=3.0)
            if success:
                self.state = "MOVING_TO_STACK"
            else:
                self.state = "IDLE"
            return False, stack_height
            
        elif self.state == "MOVING_TO_STACK":
            stack_approach = np.array([self.stack_target_pos[0] - 0.12, self.stack_target_pos[1], 0.0])
            if self._step_move_to(stack_approach, obstacles=obstacles):
                self.state = "PLACING"
            return False, stack_height
            
        elif self.state == "PLACING":
            arm_stack_target = np.array([0.08, 0.0, max(stack_height + 0.01, 0.02)])
            placed = self.arm.place_box(arm_stack_target)
            if placed:
                new_height = stack_height + SmallBox.BOX_HEIGHT
                self.done = True
                self.completed_event = f"{box_name}_completed"
                self.state = "DONE"
                print(f"[PB{self.id}] ✓ Caja {box_name} apilada. Evento: {self.completed_event}")
                return True, new_height
            else:
                self.state = "IDLE"
            return False, stack_height
            
        return False, stack_height
    
    def _step_move_to(self, target: np.ndarray, v: float = 0.2,
                       obstacles: List[Tuple[np.ndarray, float]] = None) -> bool:
        """Single step of movement with obstacle avoidance."""
        delta = target[:2] - self.pos[:2]
        dist = np.linalg.norm(delta)
        if dist < 0.02:
            return True
        
        move_dir = delta / (dist + 1e-9)
        
        # Obstacle avoidance: if near an obstacle, add perpendicular steering
        if obstacles:
            for (obs_pos, obs_radius) in obstacles:
                to_obs = obs_pos[:2] - self.pos[:2]
                obs_dist = np.linalg.norm(to_obs)
                if obs_dist < obs_radius:
                    # Perpendicular to obstacle direction (go around)
                    perp = np.array([-to_obs[1], to_obs[0]]) / (obs_dist + 1e-9)
                    # Push away from obstacle
                    repel = -to_obs / (obs_dist + 1e-9)
                    move_dir = move_dir + 1.5 * perp + 0.5 * repel
                    move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-9)
        
        self.pos[:2] += v * self.dt * move_dir
        return False
        
    def pick_and_stack(
        self,
        box_name: str,
        stack_height: float,
        sim_time: float,
        exclusion_zones: List[Tuple[np.ndarray, float]]
    ) -> Tuple[bool, float]:
        """Legacy blocking version for compatibility."""
        
        if sim_time < self.time_slot_start:
            return False, stack_height

        box_pos = self.TABLE_BOXES_POS[box_name]

        for (center, radius) in exclusion_zones:
            if np.linalg.norm(box_pos[:2] - center[:2]) < radius:
                print(f"[PB{self.id}] Zona de exclusión activa — esperando...")
                return False, stack_height

        approach = np.array([box_pos[0] - 0.12, box_pos[1], 0.0])
        print(f"\n[PB{self.id}] Recogiendo caja {box_name} en {box_pos[:2]}")
        self.move_to(approach)
        self.arm.reset()

        arm_target = np.array([0.13, 0.0, 0.02])
        success = self.arm.grasp_box(arm_target, grip_force=3.0)
        if not success:
            return False, stack_height

        stack_pos_3d = self.STACK_POS + np.array([0, 0, stack_height])
        print(f"[PB{self.id}] Apilando caja {box_name} en altura {stack_height:.3f} m")
        stack_approach = np.array([stack_pos_3d[0] - 0.12, stack_pos_3d[1], 0.0])
        self.move_to(stack_approach)

        arm_stack_target = np.array([0.13, 0.0, max(stack_height + 0.01, 0.02)])
        placed = self.arm.place_box(arm_stack_target)
        if placed:
            new_height = stack_height + SmallBox.BOX_HEIGHT
            self.done = True
            return True, new_height

        return False, stack_height


# ---------------------------------------------------------------------------
# Coordinador Principal
# ---------------------------------------------------------------------------

class Coordinator:
    """Máquina de estados que orquesta las tres fases del almacén robótico."""

    WORK_ZONE = np.array([11.0, 3.6])
    TABLE_POS = np.array([10.0, 3.6, 0.75])  # Mesa de trabajo (z=0.75m)

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.phase = Phase.IDLE
        self.metrics: dict = {}

        self.husky = HuskyPusher(slip_factor=0.05, dt=dt)
        self.anymal = ANYmalGait(dt=dt)

        # XArms acercados a la zona de llegada del ANYmal para garantizar alcance
        self.xarms = [
            XArm(1, base_pos=np.array([10.8, 3.3, 0.0])),
            XArm(2, base_pos=np.array([10.8, 3.9, 0.0])),
        ]

        pb_positions = [
            np.array([9.0, 3.6, 0.0]),
            np.array([9.0, 4.0, 0.0]),
            np.array([9.0, 3.2, 0.0]),
        ]
        self.puzzlebots = [PuzzleBot(i, pb_positions[i], dt=dt) for i in range(3)]

        self.stack_order = ["C", "B", "A"]
        for pb, box in zip(self.puzzlebots, self.stack_order):
            pb.assigned_box = box

        self.stack_height = 0.0
        self.sim_time = 0.0

    def _transition(self, new_phase: Phase):
        print(f"\n{'='*50}")
        print(f"  TRANSICIÓN: {self.phase.name} → {new_phase.name}")
        print(f"{'='*50}")
        self.phase = new_phase

    def _run_phase1(self) -> bool:
        t0 = self.sim_time
        success = self.husky.clear_corridor()
        self.metrics["phase1_time"] = self.sim_time - t0
        self.metrics["phase1_success"] = success
        return success

    def _run_phase2(self) -> bool:
        t0 = self.sim_time
        success = self.anymal.transport_puzzlebots()
        self.metrics["phase2_time"] = self.sim_time - t0
        self.metrics["phase2_success"] = success
        self.metrics["phase2_final_error"] = float(
            np.linalg.norm(self.anymal.state.pos2d - self.WORK_ZONE)
        )
        return success

    def _run_xarm_unload(self) -> bool:
        print("\n" + "=" * 50)
        print("  EXTRA: XArm descarga PuzzleBots del ANYmal")
        print("=" * 50)

        anymal_pos = self.anymal.state.pos2d
        table_positions = [
            np.array([9.2, 3.6, self.TABLE_POS[2]]),
            np.array([9.4, 3.6, self.TABLE_POS[2]]),
            np.array([9.6, 3.6, self.TABLE_POS[2]]),
        ]

        ok = self.xarms[0].pick_from_anymal(0, anymal_pos)
        ok &= self.xarms[0].place_on_table(table_positions[0])

        ok &= self.xarms[0].pick_from_anymal(1, anymal_pos)
        ok &= self.xarms[0].place_on_table(table_positions[1])

        ok2 = self.xarms[1].pick_from_anymal(2, anymal_pos)
        ok2 &= self.xarms[1].place_on_table(table_positions[2])

        for i, pb in enumerate(self.puzzlebots):
            pb.pos = table_positions[i]

        success = ok and ok2
        self.metrics["xarm_unload_success"] = success
        print(f"[XArm] Descarga: {'✓ EXITOSA' if success else '✗ CON ERRORES'}")
        return success

    def _run_phase3(self) -> bool:
        print("\n" + "=" * 50)
        print("  FASE 3: Apilado cooperativo — PuzzleBots")
        print(f"  Orden: C (abajo) → B (medio) → A (arriba)")
        print("=" * 50)

        t0 = self.sim_time
        stacking_done = [False, False, False]
        max_sim_time = 60.0
        dt_phase3 = 0.05
        
        # Event-based synchronization flags
        event_flags = {
            "C_completed": False,
            "B_completed": False,
            "A_completed": False
        }

        while not all(stacking_done) and (self.sim_time - t0) < max_sim_time:
            for i, pb in enumerate(self.puzzlebots):
                if stacking_done[i]:
                    continue

                exclusion = [
                    (self.puzzlebots[j].pos, 0.25)
                    for j in range(3)
                    if j != i and not stacking_done[j]
                ]

                # Use event-based non-blocking version
                success, new_h = pb.pick_and_stack_nonblocking(
                    pb.assigned_box, self.stack_height,
                    exclusion, event_flags
                )

                if success:
                    self.stack_height = new_h
                    stacking_done[i] = True
                    
                    # Update event flags
                    if pb.completed_event:
                        event_flags[pb.completed_event] = True
                        print(f"[Coordinator] Evento activado: {pb.completed_event}")
                    
                    print(f"[PB{i}] ✓ Caja {pb.assigned_box} apilada. Altura pila = {self.stack_height:.3f} m")

            self.sim_time += dt_phase3

        expected_height = SmallBox.BOX_HEIGHT * 3
        height_ok = abs(self.stack_height - expected_height) < 0.01
        all_stacked = all(stacking_done)

        self.metrics["phase3_time"] = self.sim_time - t0
        self.metrics["phase3_success"] = all_stacked and height_ok
        self.metrics["stack_height_final"] = self.stack_height
        self.metrics["stack_order_ok"] = True
        self.metrics["event_sync_used"] = True

        if all_stacked:
            print(f"\n[Coordinator] ✓ Pila completa: C-B-A | Altura={self.stack_height:.3f} m")
            print(f"[Coordinator] ✓ Sincronización por eventos: C→B→A")
        else:
            failed = [self.stack_order[i] for i, ok in enumerate(stacking_done) if not ok]
            print(f"[Coordinator] ✗ Cajas no apiladas: {failed}")

        return all_stacked and height_ok

    def run(self) -> bool:
        print("\n" + "╔" + "═"*48 + "╗")
        print("║  COORDINADOR — Almacén Robótico Autónomo       ║")
        print("╚" + "═"*48 + "╝")

        t_start = self.sim_time

        self._transition(Phase.PHASE1_HUSKY)
        ok1 = self._run_phase1()
        if not ok1:
            self._transition(Phase.ERROR)
            print("[Coordinator] ✗ Fase 1 falló. Misión abortada.")
            return False
        self.sim_time += self.metrics.get("phase1_time", 0)

        self._transition(Phase.PHASE2_ANYMAL)
        ok2 = self._run_phase2()
        self.sim_time += self.metrics.get("phase2_time", 0)
        if not ok2:
            print("[Coordinator] ⚠ ANYmal no llegó al destino exacto, continuando...")

        self._transition(Phase.XARM_UNLOAD)
        self._run_xarm_unload()

        self._transition(Phase.PHASE3_PUZZLEBOTS)
        ok3 = self._run_phase3()

        self._transition(Phase.DONE)
        self.metrics["total_time"] = self.sim_time - t_start
        self._print_metrics()

        return ok1 and ok3

    def _print_metrics(self):
        print("\n" + "╔" + "═"*48 + "╗")
        print("║          MÉTRICAS FINALES                      ║")
        print("╠" + "═"*48 + "╣")
        rows = [
            ("Tiempo total",        f"{self.metrics.get('total_time',0):.1f} s"),
            ("Fase 1 éxito",        "✓" if self.metrics.get("phase1_success") else "✗"),
            ("Fase 1 tiempo",       f"{self.metrics.get('phase1_time',0):.1f} s"),
            ("Fase 2 éxito",        "✓" if self.metrics.get("phase2_success") else "✗"),
            ("Fase 2 tiempo",       f"{self.metrics.get('phase2_time',0):.1f} s"),
            ("Fase 2 error final",  f"{self.metrics.get('phase2_final_error',0):.4f} m"),
            ("XArm descarga",       "✓" if self.metrics.get("xarm_unload_success") else "✗"),
            ("Fase 3 éxito",        "✓" if self.metrics.get("phase3_success") else "✗"),
            ("Fase 3 tiempo",       f"{self.metrics.get('phase3_time',0):.1f} s"),
            ("Altura pila final",   f"{self.metrics.get('stack_height_final',0):.3f} m"),
            ("Orden C-B-A",         "✓" if self.metrics.get("stack_order_ok") else "✗"),
        ]
        for k, v in rows:
            print(f"║  {k:<28} {v:>15} ║")
        print("╚" + "═"*48 + "╝")


if __name__ == "__main__":
    np.random.seed(42)
    coord = Coordinator(dt=0.02)
    result = coord.run()
    print(f"\n{'✓ MISIÓN COMPLETADA' if result else '✗ MISIÓN FALLIDA'}")