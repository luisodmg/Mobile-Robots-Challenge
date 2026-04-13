"""
torque_logger.py - Utility for logging and analyzing torque data from force control

This module provides functionality to log torque data from the PuzzleBotArm
force control implementation and generate reports for the rubric requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json

class TorqueLogger:
    """Logger and analyzer for torque data from force control operations."""
    
    def __init__(self):
        self.torque_history: List[Dict] = []
        self.force_control_events: List[Dict] = []
        
    def log_torque_data(self, robot_id: str, operation: str, torques: np.ndarray, 
                       force_applied: np.ndarray, det_J: float, timestamp: float):
        """Log torque data from force control operation."""
        entry = {
            "robot_id": robot_id,
            "operation": operation,
            "timestamp": timestamp,
            "torques": torques.tolist(),
            "torque_magnitude": float(np.linalg.norm(torques)),
            "force_applied": force_applied.tolist(),
            "det_J": det_J,
            "max_torque": float(np.max(np.abs(torques))),
            "min_torque": float(np.min(np.abs(torques)))
        }
        self.torque_history.append(entry)
        
    def log_force_control_event(self, robot_id: str, event_type: str, 
                               box_name: str, details: Dict):
        """Log force control events for tracking."""
        event = {
            "robot_id": robot_id,
            "event_type": event_type,
            "box_name": box_name,
            "timestamp": len(self.force_control_events),
            "details": details
        }
        self.force_control_events.append(event)
        
    def generate_torque_report(self, output_path: str = "torque_report.json"):
        """Generate comprehensive torque report for rubric requirements."""
        report = {
            "summary": {
                "total_operations": len(self.torque_history),
                "total_events": len(self.force_control_events),
                "robots_involved": list(set(entry["robot_id"] for entry in self.torque_history))
            },
            "torque_statistics": self._calculate_torque_stats(),
            "force_control_events": self.force_control_events,
            "detailed_torque_log": self.torque_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"[TorqueLogger] Report saved to {output_path}")
        return report
        
    def _calculate_torque_stats(self) -> Dict:
        """Calculate statistical measures of torque data."""
        if not self.torque_history:
            return {}
            
        all_torques = np.array([entry["torques"] for entry in self.torque_history])
        torque_magnitudes = np.array([entry["torque_magnitude"] for entry in self.torque_history])
        
        stats = {
            "mean_torque_magnitude": float(np.mean(torque_magnitudes)),
            "max_torque_magnitude": float(np.max(torque_magnitudes)),
            "min_torque_magnitude": float(np.min(torque_magnitudes)),
            "std_torque_magnitude": float(np.std(torque_magnitudes)),
            "joint_torque_means": all_torques.mean(axis=0).tolist(),
            "joint_torque_stds": all_torques.std(axis=0).tolist(),
            "singularities_detected": sum(1 for entry in self.torque_history if entry["det_J"] < 1e-3)
        }
        
        return stats
        
    def plot_torque_analysis(self, output_path: str = "torque_analysis.png"):
        """Generate torque analysis plots for technical report."""
        if not self.torque_history:
            print("[TorqueLogger] No torque data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Análisis de Control de Fuerza - Torques (TE3002B)", fontsize=14, fontweight='bold')
        
        # Extract data
        timestamps = [entry["timestamp"] for entry in self.torque_history]
        torque_magnitudes = [entry["torque_magnitude"] for entry in self.torque_history]
        torques_array = np.array([entry["torques"] for entry in self.torque_history])
        det_J_values = [entry["det_J"] for entry in self.torque_history]
        
        # 1. Torque magnitude over time
        axes[0, 0].plot(timestamps, torque_magnitudes, 'b-', linewidth=2)
        axes[0, 0].set_title("Magnitud de Torque vs Tiempo")
        axes[0, 0].set_xlabel("Operación")
        axes[0, 0].set_ylabel("| torque | [N·m]")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Joint torques
        for i in range(3):
            axes[0, 1].plot(timestamps, torques_array[:, i], label=f'Joint {i+1}', linewidth=2)
        axes[0, 1].set_title("Torques por Articulación")
        axes[0, 1].set_xlabel("Operación")
        axes[0, 1].set_ylabel("Torque [N·m]")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Determinant of Jacobian
        axes[1, 0].semilogy(timestamps, det_J_values, 'r-', linewidth=2)
        axes[1, 0].axhline(y=1e-3, color='red', linestyle='--', alpha=0.7, label='det(J)_min')
        axes[1, 0].set_title("Determinante del Jacobiano")
        axes[1, 0].set_xlabel("Operación")
        axes[1, 0].set_ylabel("det(J)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Torque distribution
        torque_means = torques_array.mean(axis=0)
        torque_stds = torques_array.std(axis=0)
        joint_labels = ['q1 (Base)', 'q2 (Hombro)', 'q3 (Codo)']
        
        x_pos = np.arange(len(joint_labels))
        axes[1, 1].bar(x_pos, torque_means, yerr=torque_stds, capsize=5, alpha=0.7)
        axes[1, 1].set_title("Distribución de Torques por Articulación")
        axes[1, 1].set_xlabel("Articulación")
        axes[1, 1].set_ylabel("Torque Promedio [N·m]")
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(joint_labels)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[TorqueLogger] Torque analysis plot saved to {output_path}")
        
    def print_summary(self):
        """Print summary of torque data for console output."""
        if not self.torque_history:
            print("[TorqueLogger] No torque data available")
            return
            
        stats = self._calculate_torque_stats()
        
        print("\n" + "="*60)
        print("  REPORTE DE CONTROL DE FUERZA - TORQUES")
        print("="*60)
        print(f"Operaciones totales: {len(self.torque_history)}")
        print(f"Magnitud promedio de torque: {stats['mean_torque_magnitude']:.4f} N·m")
        print(f"Magnitud máxima de torque: {stats['max_torque_magnitude']:.4f} N·m")
        print(f"Singularidades detectadas: {stats['singularities_detected']}")
        print(f"Torques promedio por articulación: {np.round(stats['joint_torque_means'], 4)} N·m")
        print("="*60)

# Global instance for use across modules
torque_logger = TorqueLogger()
