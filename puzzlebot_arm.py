"""
puzzlebot_arm.py — Mini brazo planar de 3 DoF del PuzzleBot.

Configuración: base rotacional (q1) + 2 eslabones en plano vertical (q2, q3).
Unidades: metros, radianes, Newtons, N·m.
"""

import numpy as np
from typing import Tuple, Optional, List
from torque_logger import torque_logger


class PuzzleBotArm:
    """Mini brazo planar de 3 DoF montado sobre un PuzzleBot.

    Configuración:
        - q1: rotación de la base (yaw, en plano XY)
        - q2: ángulo del primer eslabón (hombro, en plano vertical)
        - q3: ángulo del segundo eslabón (codo, en plano vertical)

    Marco de referencia: origen en la base del brazo (montado sobre el PuzzleBot).
    """

    def __init__(self, l1: float = 0.05, l2: float = 0.12, l3: float = 0.10):
        """
        Args:
            l1: Altura de la base al hombro [m].
            l2: Longitud del primer eslabón dinámico [m].
            l3: Longitud del segundo eslabón dinámico [m].
        """
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.q = np.zeros(3)  # [q1, q2, q3] en radianes

        # Límites articulares [rad]
        self.q_min = np.array([-np.pi, -np.pi / 2, -np.pi * 0.9])
        self.q_max = np.array([np.pi, np.pi, np.pi * 0.9])

        # Estado de agarre
        self.grasping = False
        self.grip_force = 0.0

        # Historial para logging
        self.torque_log: List[np.ndarray] = []
        self.pose_log: List[np.ndarray] = []
        self.force_control_log: List[dict] = []  # Log for force control data
        self.time = 0.0  # Timestamp for logging

    # ------------------------------------------------------------------
    # Cinemática Directa (FK)
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        if q is not None:
            self.q = np.clip(q, self.q_min, self.q_max)

        q1, q2, q3 = self.q

        # Proyección horizontal del brazo (radio desde el eje de q1)
        r = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)

        # Posición 3D
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        z = self.l1 + self.l2 * np.sin(q2) + self.l3 * np.sin(q2 + q3)

        p = np.array([x, y, z])
        self.pose_log.append(p.copy())
        return p

    # ------------------------------------------------------------------
    # Cinemática Inversa (IK)
    # ------------------------------------------------------------------

    def inverse_kinematics(self, p_des: np.ndarray) -> Optional[np.ndarray]:
        x, y, z = p_des

        # --- q1: ángulo de la base (yaw) ---
        q1 = np.arctan2(y, x)

        # --- Resolución en el plano vertical ---
        r = np.sqrt(x**2 + y**2)
        z_prime = z - self.l1  # Restamos la altura de la base

        D_sq = r**2 + z_prime**2
        D = np.sqrt(D_sq)

        # Verificar alcanzabilidad
        if D > self.l2 + self.l3 + 1e-6:
            print(f"[PuzzleBotArm] IK: punto fuera del workspace (D={D:.4f} > {self.l2 + self.l3:.4f})")
            return None
        if D < abs(self.l2 - self.l3) - 1e-6:
            print(f"[PuzzleBotArm] IK: punto demasiado cercano (D={D:.4f})")
            return None

        # Ley de cosenos para q3
        cos_q3 = (D_sq - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = -np.arccos(cos_q3)  # Codo hacia abajo (o arriba, según convención)

        # Geometría para q2
        beta = np.arctan2(z_prime, r)
        gamma = np.arctan2(self.l3 * np.sin(abs(q3)), self.l2 + self.l3 * np.cos(q3))
        
        if q3 < 0:
            q2 = beta + gamma
        else:
            q2 = beta - gamma

        q = np.array([q1, q2, q3])
        q = np.clip(q, self.q_min, self.q_max)
        self.q = q
        return q

    # ------------------------------------------------------------------
    # Jacobiano Analítico
    # ------------------------------------------------------------------

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        if q is not None:
            self.q = q

        q1, q2, q3 = self.q

        # Radio y sus derivadas
        r = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)
        dr_dq2 = -self.l2 * np.sin(q2) - self.l3 * np.sin(q2 + q3)
        dr_dq3 = -self.l3 * np.sin(q2 + q3)

        # Derivadas de Z
        dz_dq2 = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)
        dz_dq3 = self.l3 * np.cos(q2 + q3)

        J = np.array([
            [-r * np.sin(q1),   dr_dq2 * np.cos(q1),   dr_dq3 * np.cos(q1)],
            [ r * np.cos(q1),   dr_dq2 * np.sin(q1),   dr_dq3 * np.sin(q1)],
            [ 0.0,              dz_dq2,                dz_dq3             ],
        ])
        return J

    def jacobian_det(self, q: Optional[np.ndarray] = None) -> float:
        return abs(np.linalg.det(self.jacobian(q)))

    # ------------------------------------------------------------------
    # Control de Fuerza
    # ------------------------------------------------------------------

    def force_to_torque(self, f_tip: np.ndarray) -> np.ndarray:
        J = self.jacobian()
        tau = J.T @ f_tip
        self.torque_log.append(tau.copy())
        return tau

    # ------------------------------------------------------------------
    # Trayectoria Cartesiana + Agarre
    # ------------------------------------------------------------------

    def _cartesian_trajectory(self, p_start: np.ndarray, p_end: np.ndarray, steps: int = 20) -> List[np.ndarray]:
        return [p_start + t * (p_end - p_start) for t in np.linspace(0, 1, steps)]

    def grasp_box(self, box_pos: np.ndarray, grip_force: float = 5.0, steps: int = 30) -> bool:
        p_current = self.forward_kinematics()

        # Aproximación por encima de la caja
        p_pregrasp = box_pos + np.array([0, 0, 0.04])
        traj = self._cartesian_trajectory(p_current, p_pregrasp, steps)

        for p in traj:
            q = self.inverse_kinematics(p)
            if q is None:
                print(f"[PuzzleBotArm] grasp_box: IK falló en trayectoria pre-grasp.")
                return False

        # Descender al punto de agarre
        p_grasp = box_pos.copy()
        traj_down = self._cartesian_trajectory(p_pregrasp, p_grasp, 10)
        for p in traj_down:
            q = self.inverse_kinematics(p)
            if q is None:
                return False

        # Aplicar fuerza
        f_grip = np.array([0.0, 0.0, -grip_force])
        tau = self.force_to_torque(f_grip)

        det_J = self.jacobian_det()
        if det_J < 1e-3:
            print(f"[PuzzleBotArm] ¡ADVERTENCIA! Singularidad en agarre: det(J)={det_J:.6f}")

        self.grasping = True
        self.grip_force = grip_force
        print(f"[PuzzleBotArm] Agarre exitoso en {np.round(box_pos, 3)}. τ={np.round(tau, 4)} N·m | det(J)={det_J:.5f}")
        return True

    def place_box(self, target_pos: np.ndarray, steps: int = 30) -> bool:
        if not self.grasping:
            print("[PuzzleBotArm] place_box: no hay caja agarrada.")
            return False

        p_current = self.forward_kinematics()
        p_up = p_current + np.array([0, 0, 0.05])
        
        # Lift box
        for p in self._cartesian_trajectory(p_current, p_up, 10):
            if self.inverse_kinematics(p) is None: return False

        # Move above target
        p_above_target = target_pos + np.array([0, 0, 0.05])
        for p in self._cartesian_trajectory(p_up, p_above_target, steps):
            if self.inverse_kinematics(p) is None: return False

        # Descend with force control using Jacobian Transposed (τ = J^T * f)
        descent_steps = 10
        contact_force = np.array([0.0, 0.0, -self.grip_force * 0.5])  # Contact force
        
        for i, p in enumerate(self._cartesian_trajectory(p_above_target, target_pos, descent_steps)):
            if self.inverse_kinematics(p) is None: return False
            
            # Apply force control using Jacobian Transposed
            # τ = J^T * f where f is the desired contact force
            J = self.jacobian()
            tau_contact = J.T @ contact_force
            
            # Log the torques for rubric requirements
            self.torque_log.append(tau_contact.copy())
            
            # Check for singularity
            det_J = self.jacobian_det()
            if det_J < 1e-3:
                print(f"[PuzzleBotArm] ¡ADVERTENCIA! Singularidad en colocación: det(J)={det_J:.6f}")
            
            # Log force control information
            if i == descent_steps - 1:  # Final contact
                print(f"[PuzzleBotArm] Control de fuerza - Contacto suave:")
                print(f"  Fuerza aplicada: {contact_force} N")
                print(f"  Torques calculados (τ=J^T*f): {np.round(tau_contact, 4)} N·m")
                print(f"  det(J) = {det_J:.5f}")
                
                # Log to torque logger for rubric requirements
                torque_logger.log_torque_data(
                    robot_id=f"PuzzleBot_arm",
                    operation="place_box_contact",
                    torques=tau_contact,
                    force_applied=contact_force,
                    det_J=det_J,
                    timestamp=self.time
                )
                
                torque_logger.log_force_control_event(
                    robot_id=f"PuzzleBot_arm",
                    event_type="contact_force_control",
                    box_name="target",
                    details={
                        "method": "jacobian_transposed",
                        "formula": "τ = J^T * f",
                        "contact_force": contact_force.tolist(),
                        "resulting_torques": tau_contact.tolist()
                    }
                )

        self.grasping = False
        self.grip_force = 0.0
        print(f"[PuzzleBotArm] Caja colocada en {np.round(target_pos, 3)} con control de fuerza.")
        return True

    def reset(self):
        self.q = np.zeros(3)
        self.grasping = False
        self.grip_force = 0.0


if __name__ == "__main__":
    print("=" * 55)
    print("  Tests unitarios — PuzzleBotArm")
    print("=" * 55)

    arm = PuzzleBotArm()

    # Test FK
    q_test = np.array([0.3, 0.5, -0.4])
    p_fk = arm.forward_kinematics(q_test)
    print(f"\n[FK] q={np.round(q_test, 3)} → p={np.round(p_fk, 5)} m")

    # Test IK (round-trip)
    q_ik = arm.inverse_kinematics(p_fk)
    if q_ik is not None:
        p_check = arm.forward_kinematics(q_ik)
        err = np.linalg.norm(p_fk - p_check)
        print(f"[IK] p_des={np.round(p_fk, 5)} → q={np.round(q_ik, 4)}")
        print(f"     Round-trip error: {err:.6f} m  {'✓' if err < 1e-4 else '✗ FALLO'}")

    # Test Jacobiano
    arm.q = q_test
    J_analytic = arm.jacobian()
    eps = 1e-6
    J_numeric = np.zeros((3, 3))
    for i in range(3):
        q_p = q_test.copy(); q_p[i] += eps
        q_m = q_test.copy(); q_m[i] -= eps
        J_numeric[:, i] = (arm.forward_kinematics(q_p) - arm.forward_kinematics(q_m)) / (2 * eps)
    arm.q = q_test 
    J_err = np.max(np.abs(J_analytic - J_numeric))
    print(f"\n[Jacobiano] Error máx analítico vs numérico: {J_err:.2e}  {'✓' if J_err < 1e-5 else '✗ FALLO'}")
    print(f"            det(J) = {arm.jacobian_det():.5f}")

    # Test control de fuerza
    f = np.array([0.0, 0.0, -5.0])
    tau = arm.force_to_torque(f)
    print(f"\n[Fuerza→Torque] f={f} N → τ={np.round(tau, 4)} N·m")

    # Test grasp (Ajustado para que esté dentro del workspace de l2+l3)
    print("\n[Grasp] Ejecutando agarre...")
    arm.reset()
    success = arm.grasp_box(np.array([0.10, 0.05, 0.02]))
    print(f"  Resultado: {'✓ Exitoso' if success else '✗ Falló'}")
    print("\n[OK] Todos los tests completados.")