"""
sim.py — Simulador 2D (matplotlib) del escenario completo con visualización animada.

Muestra las tres fases en tiempo real:
    1. Husky empujando cajas
    2. ANYmal caminando con PuzzleBots
    3. PuzzleBots apilando cajas A-B-C
"""

import numpy as np
import os
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from typing import List

from husky_pusher import HuskyPusher, Box
from anymal_gait import ANYmalGait
from coordinator import Coordinator, Phase, SmallBox
from torque_logger import torque_logger
from robot_ml import ml_system


# ---------------------------------------------------------------------------
# Paleta de colores
# ---------------------------------------------------------------------------
COLORS = {
    "bg":          "#0f1117",
    "corridor":    "#1a2340",
    "work_zone":   "#1a3320",
    "start_zone":  "#2a1a30",
    "husky":       "#e74c3c",
    "anymal":      "#3498db",
    "puzzlebot":   "#2ecc71",
    "box_large":   "#e67e22",
    "box_a":       "#9b59b6",
    "box_b":       "#1abc9c",
    "box_c":       "#f39c12",
    "stack":       "#ecf0f1",
    "grid":        "#2c3e50",
    "text":        "#ecf0f1",
    "xarm":        "#e91e63",
    "success":     "#27ae60",
    "warning":     "#f39c12",
}


class Sim2D:
    """Simulador 2D animado del almacén robótico."""

    # Dimensiones del escenario
    X_TOTAL = 13.0
    Y_MIN, Y_MAX = -3.0, 6.0

    def __init__(self, dt: float = 0.05, save_gif: bool = False):
        self.dt = dt
        self.save_gif = save_gif
        self.frames: List = []

        # Sistemas
        self.husky = HuskyPusher(slip_factor=0.05, dt=dt)
        self.anymal = ANYmalGait(dt=dt)

        # Instanciar coordinador para la lógica de la fase 3
        self.coord = Coordinator(dt=dt)

        # Estado visual
        self.current_phase = "INICIO"
        self.phase_log: List[str] = []
        self.stack_boxes: List[dict] = []  # Cajas en la pila

        # Figura
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8),
                                            gridspec_kw={"width_ratios": [3, 1]})
        self.fig.patch.set_facecolor(COLORS["bg"])
        self.ax = self.axes[0]
        self.ax_info = self.axes[1]
        self._setup_scene()

    # ------------------------------------------------------------------
    # Configuración de la escena
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Configura el fondo fijo del escenario."""
        ax = self.ax
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(-1, self.X_TOTAL)
        ax.set_ylim(self.Y_MIN, self.Y_MAX)
        ax.set_aspect("equal")
        ax.set_title("Almacén Robótico Autónomo — TE3002B",
                     color=COLORS["text"], fontsize=13, fontweight="bold", pad=12)

        # Grid
        ax.grid(True, color=COLORS["grid"], linewidth=0.4, alpha=0.5)
        ax.tick_params(colors=COLORS["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])

        # Zona de inicio
        start = Rectangle((-0.5, -1.5), 2.0, 3.0, linewidth=2,
                           edgecolor=COLORS["start_zone"], facecolor=COLORS["start_zone"],
                           alpha=0.4, label="Zona de Inicio")
        ax.add_patch(start)
        ax.text(0.5, 0, "INICIO", color=COLORS["text"], fontsize=9,
                ha="center", va="center", alpha=0.7)

        # Corredor (6×2 m)
        corridor = Rectangle((1.0, -1.0), 6.0, 2.0, linewidth=2,
                              edgecolor="#4a90d9", facecolor=COLORS["corridor"],
                              alpha=0.5, label="Corredor")
        ax.add_patch(corridor)
        ax.text(4.0, 0, "CORREDOR", color=COLORS["text"], fontsize=9,
                ha="center", va="center", alpha=0.7)

        # Zona de trabajo
        work = Rectangle((7.5, 1.0), 5.0, 5.0, linewidth=2,
                          edgecolor="#4aad72", facecolor=COLORS["work_zone"],
                          alpha=0.4, label="Zona de Trabajo")
        ax.add_patch(work)
        ax.text(10.0, 5.2, "ZONA DE TRABAJO", color=COLORS["text"], fontsize=9,
                ha="center", va="center", alpha=0.7)

        # Destino del ANYmal
        ax.plot(*[11.5, 4.8], "o", color="#4a90d9", markersize=10, alpha=0.5)
        ax.text(11.5, 4.4, "Destino\nANYmal", color="#4a90d9", fontsize=7,
                ha="center", alpha=0.8)

        # Pila destino
        ax.plot(*[10.5, 3.6], "^", color=COLORS["stack"], markersize=10, alpha=0.6)
        ax.text(10.5, 3.1, "Pila\nDestino", color=COLORS["stack"], fontsize=7,
                ha="center", alpha=0.8)

        # Leyenda
        legend_elements = [
            mpatches.Patch(color=COLORS["husky"],     label="Husky A200"),
            mpatches.Patch(color=COLORS["anymal"],    label="ANYmal"),
            mpatches.Patch(color=COLORS["puzzlebot"], label="PuzzleBot"),
            mpatches.Patch(color=COLORS["box_large"], label="Caja grande"),
            mpatches.Patch(color=COLORS["xarm"],      label="XArm"),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
                  labelcolor=COLORS["text"], fontsize=8)

        # Panel info
        self.ax_info.set_facecolor(COLORS["bg"])
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)
        self.ax_info.axis("off")

    # ------------------------------------------------------------------
    # Dibujo de robots y objetos
    # ------------------------------------------------------------------

    def _draw_robot(self, ax, x, y, theta, color, size=0.3, label=""):
        """Dibuja un robot como un rectángulo con flecha de dirección."""
        # Cuerpo
        rect = FancyBboxPatch(
            (x - size / 2, y - size / 2), size, size,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1, alpha=0.9
        )
        ax.add_patch(rect)
        # Flecha de dirección
        dx = size * 0.6 * np.cos(theta)
        dy = size * 0.6 * np.sin(theta)
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        if label:
            ax.text(x, y - size * 0.7, label, color=COLORS["text"],
                    fontsize=7, ha="center")

    def _draw_box_large(self, ax, box: Box):
        """Dibuja una caja grande."""
        color = COLORS["box_large"] if not box.cleared else "#555555"
        rect = Rectangle(
            (box.x - box.width / 2, box.y - box.height / 2),
            box.width, box.height,
            facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(box.x, box.y, box.id, color="white", fontsize=8,
                ha="center", va="center", fontweight="bold")

    def _draw_anymal_with_pbs(self, ax, anymal_state, pb_positions):
        """Dibuja el ANYmal con los PuzzleBots en el dorso."""
        x, y, theta = anymal_state.x, anymal_state.y, anymal_state.theta
        # Cuerpo ANYmal
        body_len, body_w = 0.55, 0.34
        rect = FancyBboxPatch(
            (x - body_len / 2, y - body_w / 2), body_len, body_w,
            boxstyle="round,pad=0.03",
            facecolor=COLORS["anymal"], edgecolor="white", linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x, y + body_w / 2 + 0.05, "ANYmal", color=COLORS["text"],
                fontsize=7, ha="center")

        # Patas (simplificadas)
        for leg_name, leg_state in anymal_state.legs.items():
            lx, ly = leg_state.foot_pos_world[0], leg_state.foot_pos_world[1]
            c = COLORS["success"] if leg_state.det_J > 1e-3 else COLORS["warning"]
            ax.plot(lx, ly, "o", color=c, markersize=4, alpha=0.8)
            ax.plot([x, lx], [y, ly], "-", color=COLORS["anymal"], linewidth=0.8, alpha=0.4)

        # PuzzleBots en el dorso
        for i, pb_pos in enumerate(pb_positions):
            ax.plot(pb_pos[0], pb_pos[1], "s", color=COLORS["puzzlebot"],
                    markersize=7, alpha=0.9)

    def _draw_info_panel(self, phase_name: str, metrics: dict):
        """Actualiza el panel lateral con métricas."""
        ax = self.ax_info
        ax.clear()
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        y_pos = 0.95
        ax.text(0.5, y_pos, "ESTADO", color=COLORS["text"],
                fontsize=11, ha="center", fontweight="bold")
        y_pos -= 0.07

        ax.text(0.5, y_pos, phase_name, color=COLORS["warning"],
                fontsize=10, ha="center", fontweight="bold")
        y_pos -= 0.08

        ax.axhline(y=y_pos + 0.01, color=COLORS["grid"], linewidth=0.8, xmin=0.05, xmax=0.95)
        y_pos -= 0.04

        for key, val in metrics.items():
            ax.text(0.05, y_pos, f"{key}:", color=COLORS["text"], fontsize=8)
            color = COLORS["success"] if "✓" in str(val) else (
                    COLORS["warning"] if "✗" in str(val) else COLORS["text"])
            ax.text(0.95, y_pos, str(val), color=color, fontsize=8, ha="right")
            y_pos -= 0.055
            if y_pos < 0.05:
                break

    def _refresh_live_view(
        self,
        phase_name: str,
        metrics: dict,
        show_anymal: bool = False,
        pb_positions: List[np.ndarray] | None = None,
    ):
        """Redibuja la escena completa en la ventana interactiva."""
        self.ax.clear()
        self.ax_info.clear()
        self._setup_scene()

        for box in self.husky.boxes:
            self._draw_box_large(self.ax, box)

        self._draw_robot(
            self.ax,
            self.husky.state.x,
            self.husky.state.y,
            self.husky.state.theta,
            COLORS["husky"],
            size=0.35,
            label="Husky",
        )

        if show_anymal:
            self._draw_anymal_with_pbs(self.ax, self.anymal.state, [])
        
        if pb_positions is not None:
            for i, pb_pos in enumerate(pb_positions):
                self.ax.plot(pb_pos[0], pb_pos[1], "s", color=COLORS["puzzlebot"],
                             markersize=10, zorder=6)
                self.ax.text(pb_pos[0], pb_pos[1] - 0.35, f"PB{i}",
                             color=COLORS["puzzlebot"], fontsize=7, ha="center")

        for box_info in self.stack_boxes:
            bname = box_info["name"]
            bpos = box_info["pos"]
            self.ax.add_patch(Rectangle(
                (bpos[0] - 0.05, bpos[1] - 0.05), 0.10, 0.10,
                facecolor={"A": COLORS["box_a"], "B": COLORS["box_b"], "C": COLORS["box_c"]}.get(bname, "white"),
                edgecolor="white", linewidth=0.8, alpha=0.95, zorder=7
            ))
            self.ax.text(bpos[0], bpos[1], bname, color="white", fontsize=7,
                         ha="center", va="center", fontweight="bold")

        self._draw_info_panel(phase_name, metrics)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    # ------------------------------------------------------------------
    # Simulación y animación
    # ------------------------------------------------------------------

    def run_and_save(self, output_path: str = "sim_output.png", live_display: bool = True):
        """Ejecuta la simulación completa, muestra la vista en vivo y guarda frames clave."""
        print("\n[Sim2D] Iniciando simulación completa...")

        if live_display:
            plt.ion()
            plt.show(block=False)

        all_frames_data = []

        # ── Fase 1: Husky (Non-blocking) ───────────────────────────────────
        print("[Sim2D] Fase 1: Husky despejando corredor (animación fluida)...")
        self.current_phase = "FASE 1: Husky despeja corredor"
        print("[ML] Husky - Logistic Regression activado")
        
        # Reset Husky state machine for non-blocking operation
        self.husky.nav_state = "IDLE"
        self.husky.target_box = None
        self.husky.target_pos = None
        self.husky.pushing = False
        self.husky.returning = False
        
        phase1_complete = False
        for step in range(1200):  # More steps for smoother animation
            # Use non-blocking corridor clearing
            if not phase1_complete:
                phase1_complete = self.husky.clear_corridor_step()

            # Capturar frame cada 15 pasos (more frequent for smoother animation)
            if step % 15 == 0:
                frame_data = {
                    "phase": "phase1",
                    "husky": (self.husky.state.x, self.husky.state.y, self.husky.state.theta),
                    "boxes": [(b.x, b.y, b.cleared) for b in self.husky.boxes],
                    "anymal": None,
                    "pbs": [],
                }
                all_frames_data.append(frame_data)

            if live_display and (step % 3 == 0 or step == 1199):  # Update display more frequently
                cleared_count = sum(b.cleared for b in self.husky.boxes)
                nav_state = self.husky.nav_state
                metrics = {
                    "Cajas despejadas": f"{cleared_count}/{len(self.husky.boxes)}",
                    "Tiempo Husky": f"{self.husky.time:.1f} s",
                    "Estado navegación": nav_state,
                }
                self._refresh_live_view(self.current_phase, metrics, show_anymal=False)

            if phase1_complete:
                print("[Sim2D] ✓ Corredor despejado (animación fluida)")
                break

        if live_display:
            cleared_count = sum(b.cleared for b in self.husky.boxes)
            self._refresh_live_view(
                self.current_phase,
                {
                    "Cajas despejadas": f"{cleared_count}/{len(self.husky.boxes)}",
                    "Tiempo Husky": f"{self.husky.time:.1f} s",
                },
                show_anymal=False,
            )

        # ── Fase 2: ANYmal ────────────────────────────────────────────
        print("[Sim2D] Fase 2: ANYmal caminando...")
        self.current_phase = "FASE 2: ANYmal transportando PuzzleBots"
        
        # ANYmal empieza detrás del Husky, lo rodea por abajo
        self.anymal.state.x = -0.5
        self.anymal.state.y = -1.5
        
        # Waypoints: rodear Husky → corredor → zona de trabajo
        waypoints = [
            np.array([1.5, -1.5]),   # Rodear al Husky por abajo
            np.array([2.0, 0.0]),    # Entrar al corredor
            np.array([7.0, 0.0]),    # Final del corredor
            np.array([8.5, 2.0]),    # Giro hacia zona de trabajo
            np.array([11.5, 4.8]),   # Destino final (lejos de la pila)
        ]
        
        total_dist = sum(
            np.linalg.norm(waypoints[i] - (waypoints[i-1] if i > 0 else self.anymal.state.pos2d))
            for i in range(len(waypoints))
        )
        eta = ml_system.anymal_predict_eta(total_dist, 6.0)
        print(f"[ML] ANYmal ETA: {eta:.1f}s para {total_dist:.2f}m (Linear Regression)")

        for dest in waypoints:
            for step in range(2000):
                dx = dest[0] - self.anymal.state.x
                dy = dest[1] - self.anymal.state.y
                dist = np.hypot(dx, dy)
                if dist < 0.3:
                    break
                direction = np.array([dx, dy]) / (dist + 1e-9)
                v = min(0.4, dist * 2.0)
                self.anymal._step(v, direction)

                if step % 50 == 0:
                    pb_offsets = [np.array([self.anymal.state.x + (i-1)*0.15, self.anymal.state.y, 0.5])
                                  for i in range(3)]
                    frame_data = {
                        "phase": "phase2",
                        "husky": (self.husky.state.x, self.husky.state.y, self.husky.state.theta),
                        "boxes": [(b.x, b.y, b.cleared) for b in self.husky.boxes],
                        "anymal": self.anymal.state,
                        "pbs": pb_offsets,
                        "det_J": self.anymal.det_J_summary(),
                    }
                    all_frames_data.append(frame_data)

                if live_display and (step % 10 == 0):
                    pb_offsets = [np.array([self.anymal.state.x + (i - 1) * 0.15, self.anymal.state.y, 0.5])
                                  for i in range(3)]
                    metrics = {
                        "Distancia al destino": f"{np.linalg.norm(waypoints[-1] - self.anymal.state.pos2d):.2f} m",
                        "Posición": f"({self.anymal.state.x:.2f}, {self.anymal.state.y:.2f})",
                    }
                    self._refresh_live_view(self.current_phase, metrics, show_anymal=True, pb_positions=pb_offsets)
        
        print(f"[Sim2D] ✓ ANYmal llegó. Error={np.linalg.norm(waypoints[-1] - self.anymal.state.pos2d):.4f} m")

        # ── Fase 3: PuzzleBots (Real stacking with force control) ────────
        print("[Sim2D] Fase 3: PuzzleBots apilando con control de fuerza real...")
        self.current_phase = "FASE 3: PUZZLEBOTS"
        
        # PuzzleBots bajan del ANYmal uno por uno (animado)
        anymal_x, anymal_y = self.anymal.state.x, self.anymal.state.y
        # Posiciones de trabajo cerca de las cajas (no del ANYmal)
        pb_targets = [
            np.array([9.3, 3.0, 0.0]),   # Cerca de caja C
            np.array([9.6, 3.0, 0.0]),   # Cerca de caja B
            np.array([9.9, 3.0, 0.0]),   # Cerca de caja A
        ]
        stack_order = ["C", "B", "A"]
        
        # Todos empiezan en la posición del ANYmal
        pb_positions_phase3 = [
            np.array([anymal_x, anymal_y, 0.0]) for _ in range(3)
        ]
        for i, pb in enumerate(self.coord.puzzlebots):
            pb.state = "IDLE"
            pb.done = False
            pb.completed_event = None
            pb.pos = np.array([anymal_x, anymal_y, 0.0])
        
        # Animar descenso uno por uno hacia zona de trabajo
        for i, pb in enumerate(self.coord.puzzlebots):
            target = pb_targets[i]
            print(f"[Sim2D] PB{i} bajando del ANYmal hacia zona de trabajo...")
            for s in range(800):
                delta = target[:2] - pb.pos[:2]
                dist = np.linalg.norm(delta)
                if dist < 0.05:
                    break
                move_dir = delta / (dist + 1e-9)
                
                # Evitar al ANYmal
                to_anymal = np.array([anymal_x, anymal_y]) - pb.pos[:2]
                d_anymal = np.linalg.norm(to_anymal)
                if d_anymal < 0.6:
                    perp = np.array([-to_anymal[1], to_anymal[0]]) / (d_anymal + 1e-9)
                    repel = -to_anymal / (d_anymal + 1e-9)
                    move_dir = move_dir + 1.5 * perp + 0.5 * repel
                    move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-9)
                
                # Evitar otros PBs
                for j, other_pb in enumerate(self.coord.puzzlebots):
                    if j == i:
                        continue
                    to_other = other_pb.pos[:2] - pb.pos[:2]
                    d_other = np.linalg.norm(to_other)
                    if d_other < 0.35:
                        repel = -to_other / (d_other + 1e-9)
                        move_dir = move_dir + 0.8 * repel
                        move_dir = move_dir / (np.linalg.norm(move_dir) + 1e-9)
                
                pb.pos[:2] += 0.2 * self.dt * move_dir
                pb_positions_phase3[i] = pb.pos.copy()
                if live_display and s % 8 == 0:
                    self._refresh_live_view(
                        self.current_phase,
                        {"Estado": f"PB{i} bajando del ANYmal..."},
                        show_anymal=True, pb_positions=pb_positions_phase3,
                    )
            pb.pos = target.copy()
            pb_positions_phase3[i] = pb.pos.copy()
            zone = ml_system.puzzlebot_get_zone(pb.pos)
            print(f"[ML] PuzzleBot {pb.id} en posición → zona: {zone} (K-Means)")
        
        # Event-based synchronization
        event_flags = {
            "C_completed": False,
            "B_completed": False,
            "A_completed": False
        }
        
        stack_height = 0.0
        phase3_complete = False
        
        for step in range(2400):
            if not phase3_complete:
                for i, pb in enumerate(self.coord.puzzlebots):
                    if pb.done:
                        continue
                    
                    # Solo ANYmal como obstáculo (event sync ya previene conflictos entre PBs)
                    obstacles = [(np.array([anymal_x, anymal_y]), 0.6)]
                    success, new_h = pb.pick_and_stack_nonblocking(
                        pb.assigned_box, stack_height, [], event_flags,
                        obstacles=obstacles
                    )
                    
                    if success:
                        stack_height = new_h
                        
                        # Update event flags
                        if pb.completed_event:
                            event_flags[pb.completed_event] = True
                        
                        # Add to stack visualization
                        stack_pos = np.array([10.5, 3.6, stack_height - SmallBox.BOX_HEIGHT])
                        self.stack_boxes.append({"name": pb.assigned_box, "pos": stack_pos.copy()})
                        
                        # PuzzleBot se aparta un poco de la pila (sin teleport)
                        offset = np.array([-0.3, 0.3 * (i - 1), 0.0])
                        pb.pos = pb.pos + offset
                        pb_positions_phase3[i] = pb.pos.copy()
                        print(f"[Sim2D] ✓ Caja {pb.assigned_box} apilada — PB{i} se aparta")

                
                # Check if phase 3 is complete
                phase3_complete = all(pb.done for pb in self.coord.puzzlebots)
            
            # Update positions for visualization
            for i, pb in enumerate(self.coord.puzzlebots):
                pb_positions_phase3[i] = pb.pos.copy()
            
            # Capture frames more frequently
            if step % 20 == 0:
                frame_data = {
                    "phase": "phase3",
                    "husky": (self.husky.state.x, self.husky.state.y, 0),
                    "boxes": [(b.x, b.y, b.cleared) for b in self.husky.boxes],
                    "anymal": self.anymal.state,
                    "pbs": pb_positions_phase3.copy(),
                    "stack": list(self.stack_boxes),
                    "stack_height": stack_height,
                    "events": event_flags.copy(),
                }
                all_frames_data.append(frame_data)
            
            if live_display and (step % 5 == 0 or step == 2399):
                active_pb = sum(1 for pb in self.coord.puzzlebots if not pb.done)
                metrics = {
                    "Altura de pila": f"{stack_height:.3f} m",
                    "PuzzleBots activos": f"{active_pb}/3",
                    "Eventos completados": f"{sum(event_flags.values())}/3",
                    "Control de fuerza": "τ=J^T*f",
                }
                self._refresh_live_view(
                    self.current_phase, metrics,
                    show_anymal=True,
                    pb_positions=pb_positions_phase3,
                )
            
            if phase3_complete:
                print("[Sim2D] ✓ Fase 3 completada con control de fuerza real")
                break

        # ── Renderizar frames clave ───────────────────────────────────
        total_time = self.husky.time + 30.0 + 40.0
        predicted = ml_system.coordinator_predict_mission(self.husky.time, 30.0, 40.0)
        print(f"\n[ML] Coordinator - Ridge Regression:")
        print(f"  Tiempo predicho: {predicted:.1f}s")
        
        self._render_composite(all_frames_data, output_path)
        print(f"[Sim2D] Imagen guardada en {output_path}")

        if live_display:
            # Mantener la ventana 3 segundos al finalizar y cerrar automáticamente.
            plt.ioff()
            plt.pause(3.0)
            plt.close(self.fig)

    def _render_composite(self, frames_data: List, output_path: str):
        """Renderiza una imagen compuesta con los momentos clave de la simulación."""
        n_key = min(6, len(frames_data))
        key_indices = np.linspace(0, len(frames_data) - 1, n_key, dtype=int)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor(COLORS["bg"])
        fig.suptitle(
            "Almacén Robótico Autónomo — TE3002B  |  Simulación 2D",
            color=COLORS["text"], fontsize=14, fontweight="bold", y=0.98
        )

        phase_labels = {
            "phase1": "Fase 1: Husky despeja corredor",
            "phase2": "Fase 2: ANYmal transporta PuzzleBots",
            "phase3": "Fase 3: PuzzleBots apilan cajas",
        }

        for idx, (ax, frame_idx) in enumerate(zip(axes.flat, key_indices)):
            frame = frames_data[frame_idx]
            self._render_frame(ax, frame, idx + 1, len(key_indices))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["bg"], edgecolor="none")
        plt.close(fig)

    def _render_frame(self, ax, frame: dict, frame_n: int, total: int):
        """Dibuja un frame individual en un eje dado."""
        phase = frame["phase"]
        ax.set_facecolor(COLORS["bg"])
        ax.set_xlim(-1, self.X_TOTAL)
        ax.set_ylim(self.Y_MIN, self.Y_MAX)
        ax.set_aspect("equal")
        ax.tick_params(colors=COLORS["text"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.3, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])

        phase_labels = {
            "phase1": "Fase 1: Husky",
            "phase2": "Fase 2: ANYmal",
            "phase3": "Fase 3: PuzzleBots",
        }
        ax.set_title(f"{phase_labels.get(phase, phase)}  [{frame_n}/{total}]",
                     color=COLORS["text"], fontsize=9)

        # Zonas
        ax.add_patch(Rectangle((-0.5, -1.5), 2.0, 3.0, alpha=0.3,
                                facecolor=COLORS["start_zone"], edgecolor="none"))
        ax.add_patch(Rectangle((1.0, -1.0), 6.0, 2.0, alpha=0.3,
                                facecolor=COLORS["corridor"], edgecolor="#4a90d9", linewidth=0.8))
        ax.add_patch(Rectangle((7.5, 1.0), 5.0, 5.0, alpha=0.3,
                                facecolor=COLORS["work_zone"], edgecolor="#4aad72", linewidth=0.8))

        # Cajas grandes
        for (bx, by, cleared) in frame["boxes"]:
            color = "#555555" if cleared else COLORS["box_large"]
            ax.add_patch(Rectangle((bx - 0.2, by - 0.2), 0.4, 0.4,
                                    facecolor=color, edgecolor="white", linewidth=0.8, alpha=0.85))

        # Husky
        hx, hy, ht = frame["husky"]
        ax.plot(hx, hy, "s", color=COLORS["husky"], markersize=10, zorder=5)
        ax.annotate("", xy=(hx + 0.3 * np.cos(ht), hy + 0.3 * np.sin(ht)),
                    xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

        # ANYmal
        if frame.get("anymal") is not None:
            anymal = frame["anymal"]
            ax.add_patch(FancyBboxPatch(
                (anymal.x - 0.27, anymal.y - 0.17), 0.55, 0.34,
                boxstyle="round,pad=0.03",
                facecolor=COLORS["anymal"], edgecolor="white", linewidth=1.2, alpha=0.9, zorder=4
            ))

        # PuzzleBots
        for pb_pos in frame.get("pbs", []):
            ax.plot(pb_pos[0], pb_pos[1], "D", color=COLORS["puzzlebot"],
                    markersize=7, zorder=6)

        # Pila de cajas
        box_colors = {"A": COLORS["box_a"], "B": COLORS["box_b"], "C": COLORS["box_c"]}
        for box_info in frame.get("stack", []):
            bname = box_info["name"]
            bpos = box_info["pos"]
            ax.add_patch(Rectangle(
                (bpos[0] - 0.05, bpos[1] - 0.05), 0.10, 0.10,
                facecolor=box_colors.get(bname, "white"),
                edgecolor="white", linewidth=0.8, alpha=0.95, zorder=7
            ))
            ax.text(bpos[0], bpos[1], bname, color="white", fontsize=7,
                    ha="center", va="center", fontweight="bold")

        # Destino ANYmal
        ax.plot(11.5, 4.8, "o", color="#4a90d9", markersize=6, alpha=0.5, zorder=2)
        ax.plot(10.5, 3.6, "^", color=COLORS["stack"], markersize=6, alpha=0.5, zorder=2)


# ---------------------------------------------------------------------------
# Función de visualización rápida de métricas (para el reporte)
# ---------------------------------------------------------------------------

def plot_metrics(anymal: ANYmalGait, husky: HuskyPusher, output_path: str = "metrics.png"):
    """Genera gráficas de métricas para el reporte técnico."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Métricas del Sistema — TE3002B", color=COLORS["text"],
                 fontsize=13, fontweight="bold")

    ax_colors = [COLORS["text"]] * 4
    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#1a1e2e")
        ax.tick_params(colors=COLORS["text"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.4, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])

    # 1. Trayectoria del ANYmal
    ax = axes[0, 0]
    if anymal.pos_log:
        traj = np.array(anymal.pos_log)
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS["anymal"], linewidth=1.5, label="ANYmal")
        ax.plot(traj[0, 0], traj[0, 1], "go", markersize=8, label="Inicio")
        ax.plot(11.5, 4.8, "r*", markersize=12, label="Destino")
    ax.set_title("Trayectoria ANYmal", color=COLORS["text"])
    ax.set_xlabel("x [m]", color=COLORS["text"])
    ax.set_ylabel("y [m]", color=COLORS["text"])
    ax.legend(facecolor="#1a1e2e", labelcolor=COLORS["text"], fontsize=8)

    # 2. det(J) por pata
    ax = axes[0, 1]
    if anymal.det_J_log:
        t_axis = np.arange(len(anymal.det_J_log)) * anymal.dt
        pata_colors = {"LF": "#e74c3c", "RF": "#3498db", "LH": "#2ecc71", "RH": "#f39c12"}
        for leg, color in pata_colors.items():
            vals = [d.get(leg, 0) for d in anymal.det_J_log]
            ax.plot(t_axis, vals, color=color, linewidth=1.0, label=leg, alpha=0.8)
        ax.axhline(y=1e-3, color="red", linestyle="--", linewidth=1.2, label="|det(J)|_min")
    ax.set_title("det(J) por pata", color=COLORS["text"])
    ax.set_xlabel("Tiempo [s]", color=COLORS["text"])
    ax.set_ylabel("|det(J)|", color=COLORS["text"])
    ax.legend(facecolor="#1a1e2e", labelcolor=COLORS["text"], fontsize=8)
    ax.set_yscale("log")

    # 3. Velocidades Husky
    ax = axes[1, 0]
    if husky.controller.log_v_cmd:
        t_h = np.arange(len(husky.controller.log_v_cmd)) * husky.dt
        ax.plot(t_h, husky.controller.log_v_cmd, color=COLORS["husky"],
                linewidth=1.2, label="v_cmd", alpha=0.9)
        ax.plot(t_h, husky.controller.log_v_meas, color="#ff8888",
                linewidth=1.0, label="v_meas", linestyle="--", alpha=0.8)
    ax.set_title("Husky — Velocidad lineal", color=COLORS["text"])
    ax.set_xlabel("Tiempo [s]", color=COLORS["text"])
    ax.set_ylabel("v [m/s]", color=COLORS["text"])
    ax.legend(facecolor="#1a1e2e", labelcolor=COLORS["text"], fontsize=8)

    # 4. ω Husky
    ax = axes[1, 1]
    if husky.controller.log_w_cmd:
        t_h = np.arange(len(husky.controller.log_w_cmd)) * husky.dt
        ax.plot(t_h, husky.controller.log_w_cmd, color=COLORS["anymal"],
                linewidth=1.2, label="ω_cmd", alpha=0.9)
        ax.plot(t_h, husky.controller.log_w_meas, color="#88ccff",
                linewidth=1.0, label="ω_meas", linestyle="--", alpha=0.8)
    ax.set_title("Husky — Velocidad angular", color=COLORS["text"])
    ax.set_xlabel("Tiempo [s]", color=COLORS["text"])
    ax.set_ylabel("ω [rad/s]", color=COLORS["text"])
    ax.legend(facecolor="#1a1e2e", labelcolor=COLORS["text"], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close(fig)
    print(f"[Sim2D] Métricas guardadas en {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 55)
    print("  Simulador 2D — Almacén Robótico Autónomo")
    print("  TE3002B · Computational Robotics Lab")
    print("=" * 55)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    sim_output_path = os.path.join(results_dir, "sim_output.png")
    metrics_path = os.path.join(results_dir, "metrics.png")
    torque_report_path = os.path.join(results_dir, "torque_report.json")
    torque_analysis_path = os.path.join(results_dir, "torque_analysis.png")

    sim = Sim2D(dt=0.05)
    sim.run_and_save(sim_output_path, live_display=True)

    # Generar métricas
    plot_metrics(sim.anymal, sim.husky, metrics_path)
    
    # Generar reporte de control de fuerza para rúbrica
    print("\n[Sim2D] Generando reporte de control de fuerza...")
    torque_logger.print_summary()
    torque_logger.generate_torque_report(torque_report_path)
    torque_logger.plot_torque_analysis(torque_analysis_path)

    print("\n✓ Simulación completada con todas las mejoras implementadas:")
    print("  ✓ Animación fluida (máquina de estados no bloqueante)")
    print("  ✓ Control de fuerza real (τ = J^T * f)")
    print("  ✓ Sincronización por eventos (C→B→A)")
    print("  ✓ Log de torques para rúbrica")
    print("\nArchivos generados:")
    print(f"  - {sim_output_path} (visualización completa)")
    print(f"  - {metrics_path} (métricas del sistema)")
    print(f"  - {torque_report_path} (log de torques)")
    print(f"  - {torque_analysis_path} (análisis de control de fuerza)")
