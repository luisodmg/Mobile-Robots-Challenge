"""
run_without_ml.py — Ejecuta la simulación SIN modelos ML (modo clásico).

Este script ejecuta el Coordinator con ML deshabilitado para comparación.
"""

import numpy as np
from coordinator import Coordinator


def main():
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "SIMULACIÓN SIN MACHINE LEARNING" + " "*22 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nEjecutando en modo clásico (sin ML)...\n")
    
    # Crear coordinador SIN ML
    np.random.seed(42)
    coord = Coordinator(dt=0.02, use_ml=False)
    
    # Ejecutar misión
    result = coord.run()
    
    print(f"\n{'✓ MISIÓN COMPLETADA' if result else '✗ MISIÓN FALLIDA'}\n")
    
    return 0 if result else 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
