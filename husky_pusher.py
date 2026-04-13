"""
husky_pusher.py — Nodo que comanda el Husky A200 para empujar cajas.

Modelo: skid-steer con compensación de deslizamiento.
Sensor: LiDAR 2D simulado.
"""

import numpy as np
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# Constantes del Husky A200
# ---------------------------------------------------------------------------
HUSKY_WIDTH = 0.67      # m, separación entre centros de ruedas
HUSKY_MAX_V = 1.0       # m/s
HUSKY_MAX_W = 1.5       # rad/s


class HuskyState:
    """Estado cinemático del Husky."""

    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.v_meas = 0.0
        self.w_meas = 0.0

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    def __repr__(self):
        return (f"Husky(x={self.x:.3f}, y={self.y:.3f}, θ={np.degrees(self.theta):.1f}°, "
                f"v={self.v_cmd:.3f}, ω={self.w_cmd:.3f})")


class Box:
    """Caja grande (obstáculo) en el corredor."""

    def __init__(self, box_id: str, x: float, y: float, width: float = 0.4, height: float = 0.4):
        self.id = box_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cleared = False

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])


class LiDAR2D:
    """LiDAR 2D simulado para detección de cajas."""

    def __init__(self, range_max: float = 8.0, n_beams: int = 180, noise_std: float = 0.01):
        self.range_max = range_max
        self.n_beams = n_beams
        self.noise_std = noise_std
        self.angles = np.linspace(-np.pi / 2, np.pi / 2, n_beams)

    def scan(self, robot_state: HuskyState, boxes: List[Box]) -> np.ndarray:
        ranges = np.full(self.n_beams, self.range_max)

        for i, angle_local in enumerate(self.angles):
            angle_world = robot_state.theta + angle_local
            dx = np.cos(angle_world)
            dy = np.sin(angle_world)

            for box in boxes:
                if box.cleared:
                    continue
                d = self._ray_aabb(
                    robot_state.x, robot_state.y, dx, dy,
                    box.x - box.width / 2, box.y - box.height / 2,
                    box.x + box.width / 2, box.y + box.height / 2,
                )
                if d is not None and d < ranges[i]:
                    ranges[i] = d

        ranges += np.random.normal(0, self.noise_std, self.n_beams)
        ranges = np.clip(ranges, 0, self.range_max)
        return ranges

    @staticmethod
    def _ray_aabb(
        ox: float, oy: float, dx: float, dy: float,
        x_min: float, y_min: float, x_max: float, y_max: float
    ) -> float | None:
        t_min, t_max = 0.0, 1e9
        for o, d, lo, hi in [(ox, dx, x_min, x_max), (oy, dy, y_min, y_max)]:
            if abs(d) < 1e-10:
                if o < lo or o > hi:
                    return None
            else:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
        if t_min > t_max or t_max < 0:
            return None
        return t_min if t_min >= 0 else t_max


class SkidSteerController:
    """Controlador skid-steer con compensación de deslizamiento."""

    def __init__(self, slip_factor: float = 0.05):
        self.slip = slip_factor
        self.kp_v = 1.0
        self.kp_w = 2.0

        self.log_v_cmd: List[float] = []
        self.log_w_cmd: List[float] = []
        self.log_v_meas: List[float] = []
        self.log_w_meas: List[float] = []

    def compute(
        self, state: HuskyState, v_des: float, w_des: float, dt: float
    ) -> Tuple[float, float]:
        v_cmd = np.clip(v_des, -HUSKY_MAX_V, HUSKY_MAX_V)
        w_cmd = np.clip(w_des, -HUSKY_MAX_W, HUSKY_MAX_W)

        # Velocidades medidas (con deslizamiento)
        v_meas = v_cmd * (1 - self.slip) + np.random.normal(0, 0.005)
        w_meas = w_cmd * (1 - self.slip * 0.5) + np.random.normal(0, 0.005)

        # Compensación feedforward
        v_comp = v_cmd / (1 - self.slip + 1e-9)
        w_comp = w_cmd / (1 - self.slip * 0.5 + 1e-9)
        v_comp = np.clip(v_comp, -HUSKY_MAX_V, HUSKY_MAX_V)
        w_comp = np.clip(w_comp, -HUSKY_MAX_W, HUSKY_MAX_W)

        state.v_cmd = v_comp
        state.w_cmd = w_comp
        state.v_meas = v_meas
        state.w_meas = w_meas

        self.log_v_cmd.append(v_comp)
        self.log_w_cmd.append(w_comp)
        self.log_v_meas.append(v_meas)
        self.log_w_meas.append(w_meas)

        return v_comp, w_comp

    def report(self):
        if not self.log_v_cmd:
            return
        v_err = np.abs(np.array(self.log_v_cmd) - np.array(self.log_v_meas))
        w_err = np.abs(np.array(self.log_w_cmd) - np.array(self.log_w_meas))
        print(f"[SkidSteer] Error v: μ={v_err.mean():.4f}, σ={v_err.std():.4f} m/s")
        print(f"[SkidSteer] Error ω: μ={w_err.mean():.4f}, σ={w_err.std():.4f} rad/s")


class HuskyPusher:
    """Nodo principal del Husky A200 para empujar cajas del corredor."""

    CORRIDOR_X_MIN = 1.0
    CORRIDOR_X_MAX = 7.0
    CORRIDOR_Y_MIN = -1.0
    CORRIDOR_Y_MAX = 1.0

    def __init__(self, slip_factor: float = 0.05, dt: float = 0.05):
        self.state = HuskyState(x=0.0, y=0.0, theta=0.0)
        self.controller = SkidSteerController(slip_factor)
        self.lidar = LiDAR2D()
        self.dt = dt

        self.boxes = [
            Box("B1", x=2.5, y=0.0),
            Box("B2", x=4.0, y=0.3),
            Box("B3", x=5.5, y=-0.2),
        ]

        self.phase_log: List[str] = []
        self.time = 0.0
        
        # State machine for non-blocking operation
        self.nav_state = "IDLE"
        self.target_box = None
        self.target_pos = None
        self.pushing = False
        self.returning = False

    def detect_boxes(self) -> List[Dict]:
        ranges = self.lidar.scan(self.state, self.boxes)
        detected = []
        for i, r in enumerate(ranges):
            if r < self.lidar.range_max - 0.1:
                angle = self.state.theta + self.lidar.angles[i]
                bx = self.state.x + r * np.cos(angle)
                by = self.state.y + r * np.sin(angle)
                detected.append({"range": r, "angle": self.lidar.angles[i],
                                  "world_x": bx, "world_y": by})
        return detected

    def goto(
        self, x_goal: float, y_goal: float,
        tol_pos: float = 0.15, max_steps: int = 500,
        pushing: bool = False, current_box: Box = None
    ) -> bool:
        """Navega al punto (x_goal, y_goal) permitiendo rotación en el lugar."""
        for _ in range(max_steps):
            dx = x_goal - self.state.x
            dy = y_goal - self.state.y
            dist = np.hypot(dx, dy)

            if dist < tol_pos:
                return True

            angle_to_goal = np.arctan2(dy, dx)
            angle_err = self._wrap_angle(angle_to_goal - self.state.theta)

            # Acoplamiento de velocidad: solo avanza si está mirando hacia la meta
            if abs(angle_err) < 0.5: # Aprox 28 grados
                v_des = min(0.8 * dist, HUSKY_MAX_V)
            else:
                v_des = 0.0 # Rota en su propio eje primero

            w_des = 2.5 * angle_err

            v_cmd, w_cmd = self.controller.compute(self.state, v_des, w_des, self.dt)
            self._integrate(v_cmd, w_cmd)
            self.time += self.dt

            # Si estamos empujando, actualizar la posición de la caja empíricamente
            if pushing and current_box is not None:
                contact_distance = 0.4 # Distancia desde el centro del robot al centro de la caja
                current_box.x = self.state.x + contact_distance * np.cos(self.state.theta)
                current_box.y = self.state.y + contact_distance * np.sin(self.state.theta)

        return False

    def _integrate(self, v: float, w: float):
        self.state.theta += w * self.dt
        self.state.theta = self._wrap_angle(self.state.theta)
        self.state.x += v * np.cos(self.state.theta) * self.dt
        self.state.y += v * np.sin(self.state.theta) * self.dt

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def push_box_nonblocking(self, box: Box) -> bool:
        """Non-blocking version of push_box for smooth animation."""
        if self.nav_state == "IDLE":
            print(f"\n[HuskyPusher] Iniciando empuje de caja {box.id}")
            self.target_box = box
            self.nav_state = "POSITIONING"
            
            # Determine push direction
            push_dir_y = 1.0 if box.y >= 0 else -1.0
            push_target_y = push_dir_y * (self.CORRIDOR_Y_MAX + box.height + 0.3)
            
            # Set positioning target
            offset = 0.6
            self.target_pos = {
                "behind": np.array([box.x, box.y - (push_dir_y * offset)]),
                "push": np.array([box.x, push_target_y]),
                "home": np.array([0.5, 0.0])
            }
            return False
            
        elif self.nav_state == "POSITIONING":
            # Move to behind position
            target = self.target_pos["behind"]
            if self._step_goto(target[0], target[1], tol_pos=0.1):
                self.nav_state = "PUSHING"
                self.pushing = True
            return False
            
        elif self.nav_state == "PUSHING":
            # Push the box
            target = self.target_pos["push"]
            if self._step_goto(target[0], target[1], tol_pos=0.2, pushing=True):
                self.nav_state = "RETURNING"
                self.pushing = False
                
                # Check if box is cleared
                out = (self.target_box.y < self.CORRIDOR_Y_MIN - self.target_box.height / 2
                       or self.target_box.y > self.CORRIDOR_Y_MAX + self.target_box.height / 2)
                self.target_box.cleared = out
                
                if out:
                    print(f"[HuskyPusher] ✓ Caja {self.target_box.id} fuera del corredor")
                else:
                    print(f"[HuskyPusher] ✗ Caja {self.target_box.id} sigue en el corredor")
            return False
            
        elif self.nav_state == "RETURNING":
            # Return to home position
            target = self.target_pos["home"]
            if self._step_goto(target[0], target[1], tol_pos=0.15):
                self.nav_state = "IDLE"
                self.target_box = None
                self.target_pos = None
                return True
            return False
            
        return False
    
    def _step_goto(self, x_goal: float, y_goal: float, tol_pos: float = 0.15, 
                   pushing: bool = False) -> bool:
        """Single step of goto for non-blocking operation."""
        dx = x_goal - self.state.x
        dy = y_goal - self.state.y
        dist = np.hypot(dx, dy)

        if dist < tol_pos:
            return True

        angle_to_goal = np.arctan2(dy, dx)
        angle_err = self._wrap_angle(angle_to_goal - self.state.theta)

        # Velocity coupling
        if abs(angle_err) < 0.5:
            v_des = min(0.8 * dist, HUSKY_MAX_V)
        else:
            v_des = 0.0

        w_des = 2.5 * angle_err

        v_cmd, w_cmd = self.controller.compute(self.state, v_des, w_des, self.dt)
        self._integrate(v_cmd, w_cmd)
        self.time += self.dt

        # Update box position if pushing
        if pushing and self.target_box is not None:
            contact_distance = 0.4
            self.target_box.x = self.state.x + contact_distance * np.cos(self.state.theta)
            self.target_box.y = self.state.y + contact_distance * np.sin(self.state.theta)

        return False

    def clear_corridor_step(self) -> bool:
        """Single step of corridor clearing for non-blocking animation."""
        # Find next uncleared box
        if self.nav_state == "IDLE":
            for box in self.boxes:
                if not box.cleared:
                    self.push_box_nonblocking(box)
                    break
            else:
                # All boxes cleared, return to start
                if not self.returning:
                    self.nav_state = "RETURNING"
                    self.target_pos = {"home": np.array([0.5, 0.0])}
                    self.returning = True
                else:
                    if self._step_goto(0.5, 0.0, tol_pos=0.15):
                        self.returning = False
                        corridor_clear = all(box.cleared for box in self.boxes)
                        print(f"\n[HuskyPusher] Corredor {'✓ DESPEJADO' if corridor_clear else '✗ NO despejado'}")
                        return True
        else:
            # Continue current operation
            self.push_box_nonblocking(self.target_box)
            
        return False
        
    def clear_corridor(self) -> bool:
        """Legacy blocking version for compatibility."""
        print("\n" + "=" * 50)
        print("  FASE 1: Despeje del corredor — Husky A200")
        print("=" * 50)

        all_clear = True
        for box in self.boxes:
            success = self.push_box(box)
            self.phase_log.append(
                f"t={self.time:.1f}s | Caja {box.id}: {'DESPEJADA' if success else 'FALLO'}"
            )
            if not success:
                all_clear = False

        # Volver a zona de inicio
        self.goto(0.5, 0.0)

        print("\n[HuskyPusher] Reporte final:")
        for entry in self.phase_log:
            print(f"  {entry}")
        self.controller.report()

        corridor_clear = all(box.cleared for box in self.boxes)
        print(f"\n[HuskyPusher] Corredor {'✓ DESPEJADO' if corridor_clear else '✗ NO despejado'}")
        return corridor_clear

    def get_state(self) -> HuskyState:
        return self.state

    def get_boxes(self) -> List[Box]:
        return self.boxes


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    pusher = HuskyPusher(slip_factor=0.05, dt=0.05)
    result = pusher.clear_corridor()
    print(f"\nResultado: {'ÉXITO' if result else 'FALLO'}")
    print(f"Pose final Husky: {pusher.state}")