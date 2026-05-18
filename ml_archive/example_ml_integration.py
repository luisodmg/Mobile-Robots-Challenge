"""
example_ml_integration.py — Ejemplo de cómo integrar los modelos ML en los robots.

Este script muestra cómo usar cada modelo ML en el contexto del sistema real.
"""

import numpy as np
from ml_husky_safety import HuskySafetyClassifier
from ml_anymal_predictor import ANYmalTimePredictor
from ml_puzzlebot_zones import PuzzleBotZoneDiscovery
from ml_coordinator_predictor import CoordinatorMissionPredictor


def example_husky_integration():
    """Ejemplo: Usar ML para verificar seguridad antes de empujar cajas."""
    print("\n" + "="*70)
    print("  EJEMPLO 1: Husky con Clasificador de Seguridad ML")
    print("="*70)
    
    # 1. Inicializar y entrenar el clasificador
    classifier = HuskySafetyClassifier()
    X, y = classifier.generate_training_data(n_samples=1000)
    classifier.train(X, y, test_size=0.2)
    
    # 2. Simular situación del Husky
    print("\n[Situación] Husky quiere empujar caja B2")
    
    # Datos del robot
    robot_pos = np.array([3.5, 0.2])
    robot_theta = 0.1
    velocity = 0.6
    
    # Datos de la caja objetivo
    box_pos = np.array([4.0, 0.3])
    
    # Simular LiDAR (caja detectada a ~0.5m)
    lidar_ranges = np.full(180, 8.0)
    lidar_ranges[85:95] = 0.5  # Caja detectada al frente
    
    # 3. Consultar al clasificador ML
    is_safe, confidence = classifier.is_maneuver_safe(
        lidar_ranges, robot_pos, robot_theta, box_pos, velocity
    )
    
    print(f"\n[ML Decision]")
    print(f"  ¿Es seguro empujar? {is_safe}")
    print(f"  Confianza: {confidence:.1%}")
    
    if is_safe:
        print(f"  ✓ Proceder con maniobra de empuje")
    else:
        print(f"  ✗ DETENER - Maniobra peligrosa detectada")
        print(f"  → Reposicionar robot o reducir velocidad")
    
    # 4. Mostrar importancia de features
    print(f"\n[Feature Importance]")
    importance = classifier.get_feature_importance()
    for name, value in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {name:20s}: {value:.4f}")


def example_anymal_integration():
    """Ejemplo: Predecir ETA del ANYmal para coordinación."""
    print("\n" + "="*70)
    print("  EJEMPLO 2: ANYmal con Predictor de Tiempo ML")
    print("="*70)
    
    # 1. Inicializar y entrenar
    predictor = ANYmalTimePredictor()
    X, y = predictor.generate_training_data(n_samples=800)
    predictor.train(X, y, test_size=0.2)
    
    # 2. Simular situación del ANYmal
    print("\n[Situación] ANYmal transportando PuzzleBots al destino")
    
    current_pos = np.array([5.0, 2.0])
    goal_pos = np.array([11.0, 3.6])
    distance = np.linalg.norm(goal_pos - current_pos)
    
    print(f"  Posición actual: ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
    print(f"  Destino: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
    print(f"  Distancia restante: {distance:.2f} m")
    
    # 3. Predecir tiempo restante
    time_pred = predictor.predict_time_to_goal(
        distance_to_goal=distance,
        current_velocity=0.35,
        payload_kg=6.0,
        avg_det_J=0.12
    )
    
    print(f"\n[ML Prediction]")
    print(f"  Tiempo estimado de llegada: {time_pred:.1f} segundos")
    print(f"  ETA: ~{time_pred/60:.1f} minutos")
    
    # 4. Usar para coordinación
    print(f"\n[Coordinación]")
    print(f"  → Notificar a XArms: prepararse en {time_pred-5:.0f}s")
    print(f"  → Notificar a PuzzleBots: iniciar en {time_pred+10:.0f}s")


def example_puzzlebot_integration():
    """Ejemplo: Usar zonas descubiertas para navegación segura."""
    print("\n" + "="*70)
    print("  EJEMPLO 3: PuzzleBot con Descubrimiento de Zonas ML")
    print("="*70)
    
    # 1. Inicializar y entrenar
    zone_discovery = PuzzleBotZoneDiscovery(n_zones=4)
    X = zone_discovery.generate_workspace_data(n_samples=600)
    zone_discovery.train(X)
    
    # 2. Simular situación del PuzzleBot
    print("\n[Situación] PuzzleBot necesita ir de deploy a pickup zone")
    
    start_pos = np.array([9.0, 3.6, 0.0])
    goal_pos = np.array([9.8, 3.2, 0.0])
    
    print(f"  Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    print(f"  Goal: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
    
    # 3. Identificar zona actual
    cluster_id, zone_name = zone_discovery.predict_zone(
        position=start_pos,
        arm_extension=0.0,
        task_frequency=0.1,
        time_spent=0.3
    )
    
    print(f"\n[ML Zone Detection]")
    print(f"  Zona actual: {zone_name}")
    
    # 4. Generar path seguro
    path = zone_discovery.get_safe_navigation_path(start_pos, goal_pos)
    
    print(f"\n[ML Path Planning]")
    print(f"  Path generado con {len(path)} waypoints:")
    for i, wp in enumerate(path):
        print(f"    Waypoint {i}: ({wp[0]:.2f}, {wp[1]:.2f})")
    
    print(f"\n  → Path evita zonas de alta actividad (stacking)")
    print(f"  → Usa corredores de navegación descubiertos por K-Means")


def example_coordinator_integration():
    """Ejemplo: Predecir resultado de misión para optimización."""
    print("\n" + "="*70)
    print("  EJEMPLO 4: Coordinator con Predictor de Misión ML")
    print("="*70)
    
    # 1. Inicializar y entrenar
    predictor = CoordinatorMissionPredictor(alpha=1.0)
    X, y = predictor.generate_training_data(n_samples=800)
    predictor.train(X, y, test_size=0.2)
    
    # 2. Simular métricas de misión en progreso
    print("\n[Situación] Misión en progreso - Fase 2 completada")
    
    metrics = {
        "phase1_time": 22.5,
        "phase2_time": 31.2,
        "phase3_time": 0.0,  # Aún no completada
        "husky_slip": 0.06,
        "anymal_det_J_min": 0.09,
        "puzzlebot_stack_height": 0.0,  # Aún no apilado
        "xarm_success": 1
    }
    
    print(f"  Fase 1 completada: {metrics['phase1_time']:.1f}s")
    print(f"  Fase 2 completada: {metrics['phase2_time']:.1f}s")
    print(f"  Fase 3: En progreso...")
    
    # 3. Predecir tiempo total (estimando Fase 3)
    estimated_phase3 = 35.0
    
    total_time_pred = predictor.predict_total_time(
        phase1_time=metrics["phase1_time"],
        phase2_time=metrics["phase2_time"],
        phase3_time=estimated_phase3,
        husky_slip=metrics["husky_slip"],
        anymal_det_J_min=metrics["anymal_det_J_min"],
        puzzlebot_stack_height=0.12,  # Estimado
        xarm_success=metrics["xarm_success"]
    )
    
    print(f"\n[ML Prediction]")
    print(f"  Tiempo total estimado: {total_time_pred:.1f}s")
    print(f"  Tiempo restante estimado: {total_time_pred - metrics['phase1_time'] - metrics['phase2_time']:.1f}s")
    
    # 4. Comparar escenarios
    print(f"\n[Análisis de Escenarios]")
    
    # Escenario optimista
    time_best = predictor.predict_total_time(
        20.0, 25.0, 28.0, 0.02, 0.15, 0.15, 1
    )
    
    # Escenario pesimista
    time_worst = predictor.predict_total_time(
        28.0, 42.0, 50.0, 0.12, 0.02, 0.08, 0
    )
    
    print(f"  Mejor caso: {time_best:.1f}s")
    print(f"  Caso actual: {total_time_pred:.1f}s")
    print(f"  Peor caso: {time_worst:.1f}s")
    print(f"\n  → Misión en buen camino (dentro del rango esperado)")


def example_combined_workflow():
    """Ejemplo: Workflow completo usando todos los modelos ML."""
    print("\n" + "="*70)
    print("  EJEMPLO 5: Workflow Completo con Todos los Modelos ML")
    print("="*70)
    
    print("\n[Inicialización] Entrenando todos los modelos ML...")
    
    # Entrenar todos los modelos
    husky_clf = HuskySafetyClassifier()
    X_h, y_h = husky_clf.generate_training_data(500)
    husky_clf.train(X_h, y_h, test_size=0.2)
    print("  ✓ Husky Safety Classifier listo")
    
    anymal_pred = ANYmalTimePredictor()
    X_a, y_a = anymal_pred.generate_training_data(400)
    anymal_pred.train(X_a, y_a, test_size=0.2)
    print("  ✓ ANYmal Time Predictor listo")
    
    pb_zones = PuzzleBotZoneDiscovery(n_zones=4)
    X_p = pb_zones.generate_workspace_data(400)
    pb_zones.train(X_p)
    print("  ✓ PuzzleBot Zone Discovery listo")
    
    coord_pred = CoordinatorMissionPredictor(alpha=1.0)
    X_c, y_c = coord_pred.generate_training_data(400)
    coord_pred.train(X_c, y_c, test_size=0.2)
    print("  ✓ Coordinator Mission Predictor listo")
    
    print("\n[Simulación de Misión]")
    print("\n  FASE 1: Husky despejando corredor")
    lidar = np.full(180, 3.0)
    is_safe, conf = husky_clf.is_maneuver_safe(
        lidar, np.array([2.0, 0.0]), 0.0, np.array([2.5, 0.0]), 0.5
    )
    print(f"    ML: Maniobra {'segura' if is_safe else 'peligrosa'} (conf={conf:.2f})")
    
    print("\n  FASE 2: ANYmal transportando")
    time_anymal = anymal_pred.predict_time_to_goal(7.5, 0.4, 6.0, 0.12)
    print(f"    ML: ETA = {time_anymal:.1f}s")
    
    print("\n  FASE 3: PuzzleBots apilando")
    cluster, zone = pb_zones.predict_zone(np.array([9.8, 3.2, 0.0]), 0.8, 0.9, 0.5)
    print(f"    ML: Zona actual = {zone}")
    
    print("\n  COORDINACIÓN: Predicción de resultado")
    total_time = coord_pred.predict_total_time(20.0, 28.0, 32.0, 0.05, 0.08, 0.12, 1)
    print(f"    ML: Tiempo total estimado = {total_time:.1f}s")
    
    print("\n[Resumen]")
    print("  ✓ 4 modelos ML trabajando en conjunto")
    print("  ✓ Decisiones inteligentes en cada fase")
    print("  ✓ Coordinación optimizada con predicciones ML")


if __name__ == "__main__":
    np.random.seed(42)
    
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "EJEMPLOS DE INTEGRACIÓN ML" + " "*26 + "║")
    print("╚" + "="*68 + "╝")
    
    # Ejecutar todos los ejemplos
    example_husky_integration()
    example_anymal_integration()
    example_puzzlebot_integration()
    example_coordinator_integration()
    example_combined_workflow()
    
    print("\n" + "="*70)
    print("  ✅ Todos los ejemplos completados exitosamente")
    print("="*70)
    print("\n  Ver ML_README.md para documentación completa")
    print("  Ejecutar test_all_ml.py para validar todos los modelos\n")
