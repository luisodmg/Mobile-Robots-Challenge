# Integración de Machine Learning - Robots Móviles

## 📋 Resumen

Integración **simple y directa** de 4 modelos ML en el simulador `sim.py`:

| Robot | Modelo ML | Función |
|-------|-----------|---------|
| **Husky** | Logistic Regression | Clasificar seguridad de maniobras |
| **ANYmal** | Linear Regression | Predecir tiempo de llegada (ETA) |
| **PuzzleBot** | K-Means | Identificar zonas de trabajo |
| **Coordinator** | Ridge Regression | Predecir tiempo total de misión |

---

## 🚀 Uso

### Ejecutar simulación con ML:

```bash
python3 sim.py
```

Los modelos ML se ejecutan automáticamente durante la simulación y muestran sus predicciones en consola.

---

## 📊 Salida Esperada

Durante la simulación verás:

```
[ML] Sistema ML inicializado para 4 robots

Fase 1: Husky
[ML] ✓ SEGURA para B1 (conf=0.73)
[ML] ✓ SEGURA para B2 (conf=0.68)
[ML] ✓ SEGURA para B3 (conf=0.71)

Fase 2: ANYmal
[ML] ANYmal ETA: 38.5s para 11.57m (Linear Regression)

Fase 3: PuzzleBots
[ML] PuzzleBot 0 en zona: wait (K-Means)
[ML] PuzzleBot 1 en zona: wait (K-Means)
[ML] PuzzleBot 2 en zona: wait (K-Means)

Final:
[ML] Coordinator - Ridge Regression:
  Tiempo real: 95.3s
  Tiempo predicho: 92.1s
  Error: 3.2s
```

---

## 📁 Archivos

- **`robot_ml.py`** - Modelos ML simples (120 líneas)
- **`sim.py`** - Simulador con integración ML
- **`ml_archive/`** - Implementaciones completas archivadas

---

## 🔧 Detalles de Implementación

### 1. Husky - Logistic Regression

```python
# Clasificador de seguridad
is_safe, confidence = ml_system.husky_check_safety(
    min_lidar_range=2.5,
    velocity=0.5,
    angle_to_box=0.1
)
```

**Features:** `[min_range, velocity, angle]`  
**Output:** `(is_safe: bool, confidence: float)`

### 2. ANYmal - Linear Regression

```python
# Predictor de tiempo
eta = ml_system.anymal_predict_eta(
    distance=11.5,
    payload_kg=6.0
)
```

**Features:** `[distance, payload]`  
**Output:** `time_seconds: float`

### 3. PuzzleBot - K-Means

```python
# Identificador de zonas
zone = ml_system.puzzlebot_get_zone(
    position=np.array([9.8, 3.2])
)
```

**Zonas:** `pickup`, `stack`, `wait`  
**Output:** `zone_name: str`

### 4. Coordinator - Ridge Regression

```python
# Predictor de tiempo total
total_time = ml_system.coordinator_predict_mission(
    phase1_time=20.0,
    phase2_time=28.0,
    phase3_time=32.0
)
```

**Features:** `[t1, t2, t3]`  
**Output:** `total_time: float`

---

## 📦 Modelos Archivados

Si necesitas las implementaciones completas con entrenamiento desde cero:

```bash
cd ml_archive/
python3 test_all_ml.py  # Tests completos
python3 example_ml_integration.py  # Ejemplos de uso
```

Archivos disponibles:
- `ml_models.py` - Implementaciones desde cero
- `ml_husky_safety.py` - Logistic Regression completo
- `ml_anymal_predictor.py` - Linear Regression completo
- `ml_puzzlebot_zones.py` - K-Means completo
- `ml_coordinator_predictor.py` - Ridge Regression completo

---

## ✅ Ventajas de esta Integración

1. **Simple**: Solo 1 archivo (`robot_ml.py`)
2. **No invasiva**: `sim.py` funciona igual con o sin ML
3. **Directa**: Llamadas ML en el flujo natural de la simulación
4. **Ligera**: No requiere entrenamiento (modelos pre-configurados)

---

## 🎯 Próximos Pasos (Opcional)

Si quieres mejorar los modelos:

1. Entrenar con datos reales de la simulación
2. Ajustar pesos en `robot_ml.py`
3. Usar modelos completos de `ml_archive/`
