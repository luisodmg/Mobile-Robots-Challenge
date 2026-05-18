# Machine Learning Integration - Mobile Robots Challenge

## 📋 Resumen

Este proyecto integra **5 métodos de Machine Learning** en el sistema de robots móviles autónomos:

| Robot | Método ML | Propósito | Features | Target |
|-------|-----------|-----------|----------|--------|
| **Husky** | Logistic Regression | Clasificar seguridad de maniobras | LiDAR, posición, velocidad | is_safe {0,1} |
| **ANYmal** | Linear Regression (OLS) | Predecir tiempo al objetivo | Distancia, velocidad, payload, det(J) | time_remaining [s] |
| **PuzzleBot** | K-Means Clustering | Descubrir zonas operacionales | Posición, brazo, frecuencia | cluster_labels |
| **Coordinator** | Ridge Regression | Predecir resultado de misión | Métricas de 3 fases | total_time [s] |
| **Benchmark** | Random Forest | Modelo no lineal de referencia | Cualquier feature set | Regresión general |

---

## 🗂️ Estructura de Archivos

```
Mobile-Robots-Challenge/
├── ml_models.py                    # Implementaciones base de los 5 métodos ML
├── ml_husky_safety.py              # Logistic Regression para Husky
├── ml_anymal_predictor.py          # Linear Regression para ANYmal
├── ml_puzzlebot_zones.py           # K-Means para PuzzleBot
├── ml_coordinator_predictor.py     # Ridge Regression para Coordinator
├── test_all_ml.py                  # Suite de tests integral
└── ML_README.md                    # Esta documentación
```

---

## 🚀 Uso Rápido

### 1. Ejecutar todos los tests

```bash
python test_all_ml.py
```

Esto ejecutará los 5 modelos ML y mostrará un resumen de resultados.

### 2. Probar modelos individuales

```bash
# Husky - Logistic Regression
python ml_husky_safety.py

# ANYmal - Linear Regression
python ml_anymal_predictor.py

# PuzzleBot - K-Means
python ml_puzzlebot_zones.py

# Coordinator - Ridge Regression
python ml_coordinator_predictor.py
```

---

## 📊 Detalles de Cada Método

### 1️⃣ Husky - Logistic Regression (Clasificación Binaria)

**Archivo:** `ml_husky_safety.py`

**Propósito:** Clasificar si una maniobra de empuje es segura o peligrosa.

**Features (6):**
- `min_lidar_range`: Distancia mínima detectada por LiDAR
- `avg_lidar_range`: Distancia promedio
- `std_lidar_range`: Variabilidad del entorno
- `angle_to_box`: Ángulo relativo a la caja objetivo
- `distance_to_box`: Distancia euclidiana a la caja
- `velocity`: Velocidad actual del robot

**Target:** `is_safe` ∈ {0, 1}

**Algoritmo:**
```
σ(z) = 1 / (1 + e^(-z))
z = w^T x + b
Optimización: Gradient Descent
```

**Uso:**
```python
from ml_husky_safety import HuskySafetyClassifier

classifier = HuskySafetyClassifier()
X, y = classifier.generate_training_data(n_samples=1000)
classifier.train(X, y)

is_safe, confidence = classifier.is_maneuver_safe(
    lidar_ranges, robot_pos, robot_theta, box_pos, velocity
)
```

**Métricas esperadas:**
- Accuracy: > 85%
- Precision: > 80%
- Recall: > 80%

---

### 2️⃣ ANYmal - Linear Regression OLS (Predicción Continua)

**Archivo:** `ml_anymal_predictor.py`

**Propósito:** Predecir el tiempo restante para llegar al objetivo.

**Features (4):**
- `distance_to_goal`: Distancia euclidiana al destino [m]
- `current_velocity`: Velocidad actual [m/s]
- `payload_kg`: Masa del payload [kg]
- `avg_det_J`: Salud cinemática (promedio det(J) de las 4 patas)

**Target:** `time_remaining` [segundos]

**Algoritmo:**
```
θ = (X^T X)^(-1) X^T y
y_pred = X θ
```

**Uso:**
```python
from ml_anymal_predictor import ANYmalTimePredictor

predictor = ANYmalTimePredictor()
X, y = predictor.generate_training_data(n_samples=800)
predictor.train(X, y)

time_pred = predictor.predict_time_to_goal(
    distance_to_goal=6.0,
    current_velocity=0.4,
    payload_kg=6.0,
    avg_det_J=0.15
)
```

**Métricas esperadas:**
- R²: > 0.75
- MAE: < 3.0 segundos
- RMSE: < 5.0 segundos

**Interpretación de coeficientes:**
- `distance_to_goal` > 0: Más distancia → más tiempo ✓
- `current_velocity` < 0: Más velocidad → menos tiempo ✓
- `payload_kg` > 0: Más peso → más tiempo ✓
- `avg_det_J` < 0: Mejor cinemática → menos tiempo ✓

---

### 3️⃣ PuzzleBot - K-Means (Clustering No Supervisado)

**Archivo:** `ml_puzzlebot_zones.py`

**Propósito:** Descubrir zonas operacionales en el workspace sin etiquetas.

**Features (5):**
- `x_position`: Coordenada X en workspace
- `y_position`: Coordenada Y en workspace
- `arm_extension`: Extensión del brazo [0-1]
- `task_frequency`: Frecuencia de tareas [0-1]
- `time_spent`: Tiempo promedio en posición [normalizado]

**Output:** `cluster_labels` ∈ {0, 1, 2, 3}

**Zonas típicamente descubiertas:**
1. **Pickup Zone**: Alta frecuencia, brazo extendido, cerca de mesa
2. **Stacking Zone**: Muy alta frecuencia, brazo muy extendido
3. **Waiting Zone**: Baja frecuencia, brazo retraído, tiempo medio
4. **Navigation Corridors**: Frecuencia media, brazo retraído, poco tiempo

**Algoritmo:**
```
Lloyd's Algorithm:
1. Inicializar k centros aleatorios
2. Asignar puntos al centro más cercano
3. Actualizar centros = media de puntos asignados
4. Repetir hasta convergencia
```

**Uso:**
```python
from ml_puzzlebot_zones import PuzzleBotZoneDiscovery

zone_discovery = PuzzleBotZoneDiscovery(n_zones=4)
X = zone_discovery.generate_workspace_data(n_samples=600)
zone_discovery.train(X)

cluster_id, zone_name = zone_discovery.predict_zone(
    position=np.array([9.8, 3.2, 0.0]),
    arm_extension=0.8,
    task_frequency=0.9,
    time_spent=0.5
)
```

**Métricas esperadas:**
- Inertia: Minimizada tras convergencia
- Convergencia: < 50 iteraciones
- Zonas interpretables: 4 clusters distintos

---

### 4️⃣ Coordinator - Ridge Regression (Regularización L2)

**Archivo:** `ml_coordinator_predictor.py`

**Propósito:** Predecir tiempo total de misión con features correlacionadas.

**Features (7 - altamente correlacionadas):**
- `phase1_time`: Tiempo de Fase 1 [s]
- `phase2_time`: Tiempo de Fase 2 [s]
- `phase3_time`: Tiempo de Fase 3 [s]
- `husky_slip`: Factor de deslizamiento [0-1]
- `anymal_det_J_min`: Mínimo det(J) durante trayecto
- `puzzlebot_stack_height`: Altura final de pila [m]
- `xarm_success`: Éxito de XArm {0, 1}

**Target:** `total_mission_time` [segundos]

**Algoritmo:**
```
θ = (X^T X + λI)^(-1) X^T y
λ = alpha (parámetro de regularización)
```

**¿Por qué Ridge y no OLS?**
- Las features están **correlacionadas** (phase1_time, phase2_time, phase3_time)
- Ridge **estabiliza coeficientes** con regularización L2
- Previene **overfitting** cuando features se solapan

**Uso:**
```python
from ml_coordinator_predictor import CoordinatorMissionPredictor

predictor = CoordinatorMissionPredictor(alpha=1.0)
X, y = predictor.generate_training_data(n_samples=800)
predictor.train(X, y)

time_pred = predictor.predict_total_time(
    phase1_time=20.0, phase2_time=28.0, phase3_time=32.0,
    husky_slip=0.05, anymal_det_J_min=0.08,
    puzzlebot_stack_height=0.12, xarm_success=1
)
```

**Métricas esperadas:**
- R²: > 0.80
- MAE: < 5.0 segundos
- Magnitud de coeficientes: Menor que OLS (efecto de regularización)

---

### 5️⃣ Random Forest - Benchmark No Lineal

**Archivo:** `ml_models.py` (clase `RandomForestRegressor`)

**Propósito:** Modelo de referencia cuando relaciones no lineales son necesarias.

**Algoritmo:**
```
Ensemble de árboles de decisión:
1. Bootstrap sampling (n_estimators veces)
2. Entrenar árbol en cada muestra
3. Predicción = promedio de todos los árboles
```

**Cuándo usar:**
- Relaciones no lineales entre features y target
- Interacciones complejas entre variables
- Cuando modelos lineales (OLS, Ridge) tienen R² < 0.6

**Uso:**
```python
from ml_models import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

---

## 🧪 Testing

### Suite de Tests Completa

```bash
python test_all_ml.py
```

**Output esperado:**
```
╔══════════════════════════════════════════════════════════════════╗
║               TEST SUITE - ML MODELS INTEGRATION                 ║
╚══════════════════════════════════════════════════════════════════╝

======================================================================
  TEST 1: HUSKY - Logistic Regression (Safety Classifier)
======================================================================
✓ Datos generados: 1000 muestras
✓ Modelo entrenado: Accuracy=0.875
✓ Predicción validada: is_safe=True, confidence=0.923

... (más tests) ...

╔══════════════════════════════════════════════════════════════════╗
║                         RESUMEN FINAL                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Husky Logistic Regression                    ✓ PASS            ║
║  ANYmal Linear Regression                     ✓ PASS            ║
║  PuzzleBot K-Means                            ✓ PASS            ║
║  Coordinator Ridge Regression                 ✓ PASS            ║
║  Random Forest Benchmark                      ✓ PASS            ║
╚══════════════════════════════════════════════════════════════════╝

  Total: 5/5 tests pasados (100%)

  🎉 ¡TODOS LOS TESTS PASARON! 🎉
```

---

## 📈 Comparación de Métodos

| Método | Tipo | Complejidad | Interpretabilidad | Mejor Para |
|--------|------|-------------|-------------------|------------|
| **Linear Regression** | Supervisado | O(n·p²) | ⭐⭐⭐⭐⭐ | Relaciones lineales |
| **Logistic Regression** | Supervisado | O(n·p·k) | ⭐⭐⭐⭐ | Clasificación binaria |
| **Ridge Regression** | Supervisado | O(n·p²) | ⭐⭐⭐⭐ | Features correlacionadas |
| **K-Means** | No supervisado | O(n·k·i·p) | ⭐⭐⭐ | Descubrimiento de clusters |
| **Random Forest** | Supervisado | O(n·log(n)·t·p) | ⭐⭐ | Relaciones no lineales |

**Leyenda:**
- n = número de muestras
- p = número de features
- k = número de clusters/iteraciones
- t = número de árboles
- i = iteraciones hasta convergencia

---

## 🔧 Integración con Robots Existentes

### Ejemplo: Integrar en Husky

```python
# En husky_pusher.py
from ml_husky_safety import HuskySafetyClassifier

class HuskyPusher:
    def __init__(self, ...):
        # ... código existente ...
        self.safety_classifier = HuskySafetyClassifier()
        self._train_safety_model()
    
    def _train_safety_model(self):
        X, y = self.safety_classifier.generate_training_data(n_samples=1000)
        self.safety_classifier.train(X, y)
    
    def push_box_nonblocking(self, box: Box) -> bool:
        # Verificar seguridad con ML antes de ejecutar
        ranges = self.lidar.scan(self.state, self.boxes)
        is_safe, confidence = self.safety_classifier.is_maneuver_safe(
            ranges, self.state.pose[:2], self.state.theta,
            box.pos, self.state.v_cmd
        )
        
        if not is_safe:
            print(f"[ML] Maniobra peligrosa detectada (conf={confidence:.2f})")
            return False
        
        # ... continuar con lógica existente ...
```

---

## 📊 Generación de Datos de Entrenamiento

Todos los modelos incluyen métodos `generate_training_data()` que crean datos sintéticos basados en:

1. **Física del sistema**: Ecuaciones cinemáticas, dinámicas
2. **Heurísticas del dominio**: Reglas de seguridad, zonas típicas
3. **Ruido realista**: Gaussiano, sensores, actuadores

**Ventajas:**
- No requiere datos reales inicialmente
- Permite validar modelos antes de deployment
- Datos balanceados y controlados

**Para datos reales:**
```python
# Reemplazar generate_training_data() con:
X_real, y_real = load_from_robot_logs("mission_data.csv")
model.train(X_real, y_real)
```

---

## 🎯 Métricas de Evaluación

### Regresión (Linear, Ridge, Random Forest)
- **R²**: Proporción de varianza explicada (0-1, mayor es mejor)
- **MAE**: Error absoluto medio [unidades del target]
- **RMSE**: Raíz del error cuadrático medio [unidades del target]

### Clasificación (Logistic)
- **Accuracy**: Proporción de predicciones correctas (0-1)
- **Precision**: TP / (TP + FP) - evita falsos positivos
- **Recall**: TP / (TP + FN) - evita falsos negativos
- **Matriz de confusión**: TP, TN, FP, FN

### Clustering (K-Means)
- **Inertia**: Suma de distancias cuadradas a centros (menor es mejor)
- **Silhouette Score**: Calidad de clusters (-1 a 1, mayor es mejor)
- **Interpretabilidad**: ¿Los clusters tienen sentido en el dominio?

---

## 🚨 Troubleshooting

### Error: "Modelo no entrenado"
```python
# Solución: Entrenar antes de predecir
model.train(X, y)
model.predict(X_new)  # Ahora funciona
```

### K-Means no converge
```python
# Solución: Aumentar max_iter o cambiar inicialización
kmeans = KMeans(n_clusters=4, max_iter=200)
```

### Ridge tiene R² bajo
```python
# Solución: Ajustar alpha (regularización)
predictor = CoordinatorMissionPredictor(alpha=0.1)  # Menos regularización
```

### Logistic Regression no aprende
```python
# Solución: Ajustar learning_rate o n_iterations
classifier = LogisticRegression(n_features=6, learning_rate=0.1, n_iterations=2000)
```

---

## 📚 Referencias

- **Linear Regression**: Ordinary Least Squares (OLS)
- **Logistic Regression**: Binary classification with sigmoid
- **Ridge Regression**: L2 regularization for correlated features
- **K-Means**: Lloyd's algorithm for unsupervised clustering
- **Random Forest**: Ensemble of decision trees with bootstrap

---

## ✅ Checklist de Implementación

- [x] Implementar 5 métodos ML desde cero (sin sklearn)
- [x] Crear módulos de integración para cada robot
- [x] Generar datos sintéticos de entrenamiento
- [x] Implementar métricas de evaluación
- [x] Suite de tests integral
- [x] Documentación completa
- [ ] Integrar en clases de robots existentes (opcional)
- [ ] Visualizaciones de resultados (opcional)
- [ ] Logs de predicciones en tiempo real (opcional)

---

## 👥 Autores

Proyecto desarrollado para el curso **TE3002B - Implementation of Intelligent Robotics**

---

## 📄 Licencia

Proyecto académico - Tecnológico de Monterrey
