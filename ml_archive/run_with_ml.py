"""
run_with_ml.py — Ejecuta la simulación completa CON modelos ML integrados.

Este script ejecuta el Coordinator con ML habilitado en todos los robots.
"""

import numpy as np
from coordinator import Coordinator


def main():
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "SIMULACIÓN CON MACHINE LEARNING INTEGRADO" + " "*16 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nModelos ML que se entrenarán:")
    print("  1. Husky - Logistic Regression (Safety Classifier)")
    print("  2. ANYmal - Linear Regression (Time-to-Goal Predictor)")
    print("  3. PuzzleBot - K-Means (Zone Discovery)")
    print("  4. Coordinator - Ridge Regression (Mission Predictor)")
    print("\nEsto puede tomar unos segundos...\n")
    
    # Crear coordinador CON ML habilitado
    np.random.seed(42)
    coord = Coordinator(dt=0.02, use_ml=True)
    
    print("\n" + "="*70)
    print("  Todos los modelos ML entrenados. Iniciando simulación...")
    print("="*70)
    
    # Ejecutar misión
    result = coord.run()
    
    # Resumen final
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*25 + "RESUMEN FINAL" + " "*30 + "║")
    print("╠" + "="*68 + "╣")
    
    if result:
        print("║  Estado: ✓ MISIÓN COMPLETADA" + " "*38 + "║")
    else:
        print("║  Estado: ✗ MISIÓN FALLIDA" + " "*41 + "║")
    
    # Estadísticas ML
    if coord.use_ml:
        print("╠" + "="*68 + "╣")
        print("║" + " "*20 + "ESTADÍSTICAS ML" + " "*33 + "║")
        print("╠" + "="*68 + "╣")
        
        # Husky ML stats
        if coord.husky.use_ml:
            safe = coord.husky.ml_decisions["safe"]
            unsafe = coord.husky.ml_decisions["unsafe"]
            total = safe + unsafe
            if total > 0:
                print(f"║  Husky ML Decisions: {safe} seguras, {unsafe} peligrosas" + " "*(68-47-len(str(safe))-len(str(unsafe))) + "║")
        
        # ANYmal ML stats
        if coord.anymal.use_ml and coord.anymal.ml_predictions:
            avg_eta = np.mean(coord.anymal.ml_predictions)
            print(f"║  ANYmal ML ETA promedio: {avg_eta:.1f}s" + " "*(68-36-len(f"{avg_eta:.1f}")) + "║")
        
        # Coordinator ML stats
        if coord.ml_mission_predictor:
            print(f"║  Mission Predictor: Activo" + " "*41 + "║")
        
        if coord.ml_zone_discovery:
            centers = coord.ml_zone_discovery.get_zone_centers()
            print(f"║  Zone Discovery: {len(centers)} zonas descubiertas" + " "*(68-44-len(str(len(centers)))) + "║")
    
    print("╚" + "="*68 + "╝")
    
    print(f"\n{'✓ SIMULACIÓN COMPLETADA CON ML' if result else '✗ SIMULACIÓN FALLIDA'}\n")
    
    return 0 if result else 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
