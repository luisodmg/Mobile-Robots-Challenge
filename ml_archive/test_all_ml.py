"""
test_all_ml.py — Script de prueba integral para todos los modelos ML.

Ejecuta y valida los 5 métodos de ML implementados:
1. Linear Regression (OLS) - ANYmal
2. Logistic Regression - Husky
3. K-Means - PuzzleBot
4. Ridge Regression - Coordinator
5. Random Forest - Benchmark (opcional)
"""

import numpy as np
import sys


def test_husky_logistic_regression():
    """Test del clasificador de seguridad del Husky."""
    print("\n" + "="*70)
    print("  TEST 1: HUSKY - Logistic Regression (Safety Classifier)")
    print("="*70)
    
    from ml_husky_safety import HuskySafetyClassifier
    
    classifier = HuskySafetyClassifier()
    
    # Generar y entrenar
    X, y = classifier.generate_training_data(n_samples=1000)
    print(f"✓ Datos generados: {len(X)} muestras")
    
    test_acc = classifier.train(X, y, test_size=0.2)
    print(f"✓ Modelo entrenado: Accuracy={test_acc:.3f}")
    
    # Validar predicción
    lidar_safe = np.full(180, 5.0)
    is_safe, conf = classifier.is_maneuver_safe(
        lidar_safe,
        robot_pos=np.array([2.0, 0.0]),
        robot_theta=0.0,
        box_pos=np.array([3.5, 0.1]),
        velocity=0.4
    )
    
    assert is_safe == True, "Debería clasificar como seguro"
    assert conf > 0.5, "Confianza debería ser > 0.5"
    print(f"✓ Predicción validada: is_safe={is_safe}, confidence={conf:.3f}")
    
    return True


def test_anymal_linear_regression():
    """Test del predictor de tiempo del ANYmal."""
    print("\n" + "="*70)
    print("  TEST 2: ANYMAL - Linear Regression OLS (Time-to-Goal Predictor)")
    print("="*70)
    
    from ml_anymal_predictor import ANYmalTimePredictor
    
    predictor = ANYmalTimePredictor()
    
    # Generar y entrenar
    X, y = predictor.generate_training_data(n_samples=800)
    print(f"✓ Datos generados: {len(X)} muestras")
    
    r2_test = predictor.train(X, y, test_size=0.2)
    print(f"✓ Modelo entrenado: R²={r2_test:.3f}")
    
    # Validar predicción
    time_pred = predictor.predict_time_to_goal(
        distance_to_goal=6.0,
        current_velocity=0.4,
        payload_kg=6.0,
        avg_det_J=0.15
    )
    
    assert time_pred > 0, "Tiempo predicho debe ser positivo"
    assert 5.0 < time_pred < 50.0, "Tiempo debe estar en rango razonable"
    print(f"✓ Predicción validada: time={time_pred:.2f} s")
    
    # Validar coeficientes
    coeffs = predictor.get_coefficients()
    assert "distance_to_goal" in coeffs, "Debe tener coeficiente de distancia"
    print(f"✓ Coeficientes extraídos: {len(coeffs)} features")
    
    return True


def test_puzzlebot_kmeans():
    """Test del descubridor de zonas del PuzzleBot."""
    print("\n" + "="*70)
    print("  TEST 3: PUZZLEBOT - K-Means (Workspace Zone Discovery)")
    print("="*70)
    
    from ml_puzzlebot_zones import PuzzleBotZoneDiscovery
    
    zone_discovery = PuzzleBotZoneDiscovery(n_zones=4)
    
    # Generar y entrenar
    X = zone_discovery.generate_workspace_data(n_samples=600)
    print(f"✓ Datos generados: {len(X)} observaciones")
    
    labels = zone_discovery.train(X)
    print(f"✓ Modelo entrenado: {zone_discovery.n_zones} zonas descubiertas")
    
    # Validar predicción
    cluster_id, zone_name = zone_discovery.predict_zone(
        position=np.array([9.8, 3.2, 0.0]),
        arm_extension=0.8,
        task_frequency=0.9,
        time_spent=0.5
    )
    
    assert 0 <= cluster_id < 4, "Cluster ID debe estar en rango"
    assert zone_name in zone_discovery.zone_names.values(), "Nombre de zona debe existir"
    print(f"✓ Predicción validada: cluster={cluster_id}, zone={zone_name}")
    
    # Validar centros
    centers = zone_discovery.get_zone_centers()
    assert len(centers) > 0, "Debe tener centros de zonas"
    print(f"✓ Centros de zonas: {len(centers)} zonas identificadas")
    
    return True


def test_coordinator_ridge_regression():
    """Test del predictor de misión del Coordinator."""
    print("\n" + "="*70)
    print("  TEST 4: COORDINATOR - Ridge Regression (Mission Predictor)")
    print("="*70)
    
    from ml_coordinator_predictor import CoordinatorMissionPredictor
    
    predictor = CoordinatorMissionPredictor(alpha=1.0)
    
    # Generar y entrenar
    X, y = predictor.generate_training_data(n_samples=800)
    print(f"✓ Datos generados: {len(X)} misiones")
    
    r2_test = predictor.train(X, y, test_size=0.2)
    print(f"✓ Modelo entrenado: R²={r2_test:.3f}")
    
    # Validar predicción
    time_pred = predictor.predict_total_time(
        phase1_time=20.0,
        phase2_time=28.0,
        phase3_time=32.0,
        husky_slip=0.05,
        anymal_det_J_min=0.08,
        puzzlebot_stack_height=0.12,
        xarm_success=1
    )
    
    assert time_pred > 0, "Tiempo predicho debe ser positivo"
    assert 50.0 < time_pred < 150.0, "Tiempo total debe estar en rango razonable"
    print(f"✓ Predicción validada: total_time={time_pred:.1f} s")
    
    # Validar regularización
    coeffs = predictor.get_coefficients()
    assert coeffs["alpha"] == 1.0, "Alpha debe ser 1.0"
    print(f"✓ Regularización L2: α={coeffs['alpha']}")
    
    return True


def test_random_forest_benchmark():
    """Test del Random Forest (benchmark no lineal)."""
    print("\n" + "="*70)
    print("  TEST 5: RANDOM FOREST - Nonlinear Benchmark (Optional)")
    print("="*70)
    
    from ml_models import RandomForestRegressor
    
    # Generar datos no lineales
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (500, 3))
    y = np.sin(X[:, 0]) + X[:, 1]**2 + 0.5 * X[:, 2] + np.random.normal(0, 0.1, 500)
    
    print(f"✓ Datos no lineales generados: {len(X)} muestras")
    
    # Entrenar
    rf = RandomForestRegressor(n_estimators=10, max_depth=5)
    r2 = rf.fit(X, y)
    print(f"✓ Random Forest entrenado: R²={r2:.3f}")
    
    # Validar predicción
    X_test = np.array([[1.0, 0.5, -0.5]])
    y_pred = rf.predict(X_test)
    
    assert len(y_pred) == 1, "Debe predecir 1 valor"
    print(f"✓ Predicción validada: y_pred={y_pred[0]:.3f}")
    
    return True


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "TEST SUITE - ML MODELS INTEGRATION" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Husky Logistic Regression", test_husky_logistic_regression),
        ("ANYmal Linear Regression", test_anymal_linear_regression),
        ("PuzzleBot K-Means", test_puzzlebot_kmeans),
        ("Coordinator Ridge Regression", test_coordinator_ridge_regression),
        ("Random Forest Benchmark", test_random_forest_benchmark),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "✓ PASS"))
            print(f"\n{'='*70}")
            print(f"  {name}: ✓ PASS")
            print(f"{'='*70}")
        except Exception as e:
            results.append((name, f"✗ FAIL: {str(e)}"))
            print(f"\n{'='*70}")
            print(f"  {name}: ✗ FAIL")
            print(f"  Error: {str(e)}")
            print(f"{'='*70}")
    
    # Resumen final
    print("\n\n" + "╔" + "="*68 + "╗")
    print("║" + " "*25 + "RESUMEN FINAL" + " "*30 + "║")
    print("╠" + "="*68 + "╣")
    
    for name, result in results:
        status = result.split(":")[0]
        print(f"║  {name:45s} {status:20s} ║")
    
    print("╚" + "="*68 + "╝")
    
    # Estadísticas
    passed = sum(1 for _, r in results if "PASS" in r)
    total = len(results)
    
    print(f"\n  Total: {passed}/{total} tests pasados ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n  🎉 ¡TODOS LOS TESTS PASARON! 🎉\n")
        return 0
    else:
        print("\n  ⚠️  Algunos tests fallaron. Revisa los errores arriba.\n")
        return 1


if __name__ == "__main__":
    np.random.seed(42)
    exit_code = run_all_tests()
    sys.exit(exit_code)
