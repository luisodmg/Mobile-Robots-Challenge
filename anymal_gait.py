"""
anymal_gait.py — Generador de marcha trote del ANYmal con FK/IK por pata.

Monitorea det(J) para evitar singularidades.
Payload: 3 PuzzleBots ≈ 6 kg.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from robot_ml_policy import get_robot_policy


# ---------------------------------------------------------------------------
# Parámetros del ANYmal
# ---------------------------------------------------------------------------
ANYMAL_BODY_LENGTH = 0.55   # m
ANYMAL_BODY_WIDTH  = 0.34   # m
ANYMAL_LEG_OFFSETS = {
    "LF": np.array([ ANYMAL_BODY_LENGTH / 2,  ANYMAL_BODY_WIDTH / 2, 0.0]),
    "RF": np.array([ ANYMAL_BODY_LENGTH / 2, -ANYMAL_BODY_WIDTH / 2, 0.0]),
    "LH": np.array([-ANYMAL_BODY_LENGTH / 2,  ANYMAL_BODY_WIDTH / 2, 0.0]),
    "RH": np.array([-ANYMAL_BODY_LENGTH / 2, -ANYMAL_BODY_WIDTH / 2, 0.0]),
}

L_HAA = 0.0   # offset lateral (simplificado)
L_HFE = 0.20  # fémur
L_KFE = 0.21  # tibia

H_NOMINAL = 0.42   # Altura nominal

# Singularidad: |det(J)| mínimo permitido
DET_J_MIN = 1e-3


@dataclass
class LegState:
    name: str
    q: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.6, -1.4]))
    foot_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    in_contact: bool = True
    det_J: float = 1.0


@dataclass
class ANYmalState:
    x: float = 0.0
    y: float = 0.0
    z: float = H_NOMINAL
    theta: float = 0.0
    legs: dict = field(default_factory=dict)
    payload_kg: float = 6.0

    @property
    def pos2d(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def pose3d(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class LegKinematics:
    def __init__(self, leg_name: str, body_offset: np.ndarray):
        self.name = leg_name
        self.body_offset = body_offset
        self.l_hfe = L_HFE
        self.l_kfe = L_KFE
        self.sign_lat = 1.0 if "L" in leg_name else -1.0

    def forward_kinematics(self, q: np.ndarray, body_pos: np.ndarray, body_yaw: float) -> np.ndarray:
        q_haa, q_hfe, q_kfe = q

        # Posición local
        x_leg = self.l_hfe * np.sin(q_hfe) + self.l_kfe * np.sin(q_hfe + q_kfe)
        y_leg = self.sign_lat * L_HAA 
        z_leg = -(self.l_hfe * np.cos(q_hfe) + self.l_kfe * np.cos(q_hfe + q_kfe))

        foot_hip = np.array([x_leg, y_leg, z_leg])

        R = np.array([
            [np.cos(body_yaw), -np.sin(body_yaw), 0],
            [np.sin(body_yaw),  np.cos(body_yaw), 0],
            [0, 0, 1]
        ])

        return body_pos + R @ (self.body_offset + foot_hip)

    def inverse_kinematics(self, foot_world: np.ndarray, body_pos: np.ndarray, body_yaw: float) -> Optional[np.ndarray]:
        R = np.array([
            [np.cos(body_yaw), -np.sin(body_yaw), 0],
            [np.sin(body_yaw),  np.cos(body_yaw), 0],
            [0, 0, 1]
        ])
        foot_hip = R.T @ (foot_world - body_pos) - self.body_offset

        x, y, z = foot_hip

        q_haa = np.arctan2(y * self.sign_lat, -z + 1e-9)

        r = np.sqrt(x**2 + z**2)

        # Límite matemático para evitar singularidad (|det(J)| > 1e-3)
        cos_kfe = (r**2 - self.l_hfe**2 - self.l_kfe**2) / (2 * self.l_hfe * self.l_kfe)
        # Limitamos a 0.995 en vez de 1.0 para forzar que la rodilla nunca se estire a 0 rad
        cos_kfe = np.clip(cos_kfe, -1.0, 0.995) 
        
        q_kfe = -np.arccos(cos_kfe)  # Rodilla siempre flexionada

        beta = np.arctan2(-z, x)
        gamma = np.arctan2(self.l_kfe * np.sin(-q_kfe), self.l_hfe + self.l_kfe * np.cos(-q_kfe))
        q_hfe = beta - gamma

        return np.array([q_haa, q_hfe, q_kfe])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        # Jacobiano numérico consistente con la FK
        body_ref = np.zeros(3)
        yaw_ref = 0.0
        eps = 1e-5
        J = np.zeros((3, 3))
        for i in range(3):
            qp = q.copy(); qp[i] += eps
            qm = q.copy(); qm[i] -= eps
            fp = self.forward_kinematics(qp, body_ref, yaw_ref)
            fm = self.forward_kinematics(qm, body_ref, yaw_ref)
            J[:, i] = (fp - fm) / (2 * eps)
        return J

    def jacobian_det(self, q: np.ndarray) -> float:
        return abs(np.linalg.det(self.jacobian(q)))


class TrotGait:
    """Generador de marcha trote dinámico relativo al cuerpo."""

    def __init__(self, step_height: float = 0.08, step_duration: float = 0.5, dt: float = 0.01):
        self.step_height = step_height
        self.T = step_duration
        self.dt = dt
        self.phase = 0.0 

        self.diagonal_pairs = [("LF", "RH"), ("RF", "LH")]

    def advance(self, v_body: float) -> dict:
        """Calcula los offsets de los pies en el marco local del cuerpo."""
        self.phase = (self.phase + self.dt / self.T) % 1.0
        
        # El paso longitudinal depende de la velocidad del robot y el tiempo de apoyo (T/2)
        stride = v_body * (self.T / 2.0)
        offsets = {}

        for i, (leg_a, leg_b) in enumerate(self.diagonal_pairs):
            phi_leg = (self.phase + i * 0.5) % 1.0

            if phi_leg < 0.5:
                # Fase de vuelo (Swing): Mover pie hacia adelante
                t = phi_leg * 2.0 
                x = -stride / 2.0 + stride * t
                z = self.step_height * np.sin(np.pi * t)
            else:
                # Fase de apoyo (Stance): Mover pie hacia atrás relativo al cuerpo
                t = (phi_leg - 0.5) * 2.0
                x = stride / 2.0 - stride * t
                z = 0.0

            offset = np.array([x, 0.0, z])
            offsets[leg_a] = offset
            offsets[leg_b] = offset

        return offsets


class ANYmalGait:
    DEST = np.array([11.0, 3.6])  # Zona de trabajo

    def __init__(self, dt: float = 0.01):
        self.state = ANYmalState()
        self.dt = dt
        self.gait = TrotGait(dt=dt)
        self.ml_policy = get_robot_policy()

        self.leg_kin = {
            name: LegKinematics(name, offset)
            for name, offset in ANYMAL_LEG_OFFSETS.items()
        }

        self._init_feet()

        self.det_J_log: List[dict] = []
        self.pos_log: List[np.ndarray] = []
        self.singularity_events: List[dict] = []

    def _init_feet(self):
        for name, kin in self.leg_kin.items():
            q_init = np.array([0.0, 0.6, -1.4])
            foot = kin.forward_kinematics(q_init, self.state.pose3d, self.state.theta)
            self.state.legs[name] = LegState(name=name, q=q_init, foot_pos_world=foot)

    def _update_jacobians(self) -> bool:
        safe = True
        dets = {}
        for name, leg in self.state.legs.items():
            kin = self.leg_kin[name]
            d = kin.jacobian_det(leg.q)
            leg.det_J = d
            dets[name] = d
            if d < DET_J_MIN:
                safe = False
                event = {"t": len(self.det_J_log) * self.dt, "leg": name, "det_J": d}
                self.singularity_events.append(event)
        
        self.det_J_log.append(dets)
        return safe

    def _step(self, v: float, direction: np.ndarray):
        # 1. Integrar cuerpo
        self.state.x += v * direction[0] * self.dt
        self.state.y += v * direction[1] * self.dt
        self.state.theta = np.arctan2(direction[1], direction[0])

        R_body = np.array([
            [np.cos(self.state.theta), -np.sin(self.state.theta), 0],
            [np.sin(self.state.theta),  np.cos(self.state.theta), 0],
            [0, 0, 1]
        ])

        # 2. Obtener offsets locales (relativos al avance)
        offsets_local = self.gait.advance(v)

        # 3. Aplicar IK
        for name, leg in self.state.legs.items():
            kin = self.leg_kin[name]
            off_local = offsets_local[name]
            
            # Convertir el offset local a coordenadas del mundo sumándolo al offset de cadera
            foot_des_world = self.state.pose3d + R_body @ (ANYMAL_LEG_OFFSETS[name] + off_local)

            q = kin.inverse_kinematics(foot_des_world, self.state.pose3d, self.state.theta)
            if q is not None:
                leg.q = q
                leg.foot_pos_world = kin.forward_kinematics(q, self.state.pose3d, self.state.theta)

        self._update_jacobians()
        self.pos_log.append(self.state.pos2d.copy())

    def walk_to(self, x_goal: float, y_goal: float, v: float = 0.4, tol: float = 0.15) -> bool:
        max_steps = int(60 / self.dt)
        for step_i in range(max_steps):
            dx = x_goal - self.state.x
            dy = y_goal - self.state.y
            dist = np.hypot(dx, dy)

            if dist < tol:
                print(f"[ANYmal] ✓ Llegué a destino. Error={dist:.4f} m < tol={tol} m")
                return True

            direction = np.array([dx, dy]) / (dist + 1e-9)
            det_j_min = min((leg.det_J for leg in self.state.legs.values()), default=1.0)
            speed_scale, step_gain = self.ml_policy.anymal_speed_profile(
                dist=dist,
                det_j_min=det_j_min,
                payload_kg=self.state.payload_kg,
            )
            v_current = min(v * speed_scale * step_gain, dist * 2.0)
            self._step(v_current, direction)

            if step_i % 500 == 0:
                print(
                    f"[ANYmal] t={step_i*self.dt:.1f}s | pos=({self.state.x:.2f},{self.state.y:.2f}) | "
                    f"dist={dist:.2f}m | ML x{speed_scale:.2f}"
                )

        dist_final = np.hypot(x_goal - self.state.x, y_goal - self.state.y)
        print(f"[ANYmal] ✗ Tiempo agotado. Error final={dist_final:.4f} m")
        return dist_final < tol

    def transport_puzzlebots(self) -> bool:
        print("\n" + "=" * 50)
        print("  FASE 2: Transporte ANYmal (trote)")
        print(f"  Payload: {self.state.payload_kg} kg  |  Destino: {self.DEST}")
        print("=" * 50)

        success = self.walk_to(self.DEST[0], self.DEST[1])

        n_sing = len(self.singularity_events)
        if n_sing == 0:
            print("[ANYmal] ✓ Sin singularidades durante el recorrido.")
        else:
            print(f"[ANYmal] ⚠ {n_sing} eventos de singularidad detectados:")
            for ev in self.singularity_events[:5]:
                print(f"   t={ev['t']:.2f}s | pata={ev['leg']} | det(J)={ev['det_J']:.2e}")

        print(f"[ANYmal] Pose final: ({self.state.x:.3f}, {self.state.y:.3f})")
        return success

    def det_J_summary(self) -> dict:
        if not self.det_J_log:
            return {}
        summary = {}
        for name in self.state.legs:
            vals = [d[name] for d in self.det_J_log]
            summary[name] = {
                "min": float(np.min(vals)),
                "mean": float(np.mean(vals)),
                "violations": int(np.sum(np.array(vals) < DET_J_MIN)),
            }
        return summary


if __name__ == "__main__":
    np.random.seed(0)
    robot = ANYmalGait(dt=0.02)
    result = robot.transport_puzzlebots()

    print("\n[Resumen det(J)]")
    for leg, stats in robot.det_J_summary().items():
        print(f"  {leg}: min={stats['min']:.4f} | mean={stats['mean']:.4f} | violaciones={stats['violations']}")

    print(f"\nResultado Fase 2: {'ÉXITO' if result else 'FALLO'}")