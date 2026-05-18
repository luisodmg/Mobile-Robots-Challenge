# IRS Inc. AI Hackathon 2026 - Navegación Multi-Robot con Visión

## 🎯 Objetivo del Hackathon

Demostrar que una flota de **N ≥ 2 robots móviles** puede realizar tareas logísticas en una planta de manufactura simulada usando **únicamente visión por computadora**, sin LiDAR, sonar, GPS ideal ni telemetría láser.

---

## ✅ Requisitos Cumplidos

### 1. Multi-Robot (N≥2) ✓
- **3 PuzzleBots** con brazo 3 DoF
- **1 Husky A200** para despeje de corredor
- **1 ANYmal** cuadrúpedo para transporte

### 2. Mapa Conocido ✓
- Planta industrial simulada 2D
- **M = 3** estaciones de pick-up (cajas A, B, C)
- **K = 1** punto de drop-off (pila destino)
- Obstáculos estáticos conocidos

### 3. Percepción Visual RGB ✓
**Cada robot tiene una cámara RGB monocular simulada**

#### Técnicas de Visión Implementadas:

**Técnica 1: Detección por Color**
- Clasificación de objetos por canal RGB dominante
- Detección de cajas grandes (naranja)
- Detección de robots (rojo, azul, verde)
- Implementado en: `vision_perception.py::detect_by_color()`

**Técnica 2: Detección de Contornos**
- Análisis de bounding boxes
- Filtrado por área (50-10000 píxeles²)
- Estimación de tamaño desde contorno
- Implementado en: `vision_perception.py::detect_contours()`

**Técnica 3: Landmarks ArUco (Opcional)**
- 4 marcadores ArUco en posiciones clave
- Localización visual relativa
- Implementado en: `vision_perception.py::detect_aruco_landmarks()`

**Técnica 4: Estimación de Distancia**
- Cálculo desde tamaño en píxeles
- Fórmula: `distance = (size_real * focal_length) / size_pixels`
- Implementado en: `vision_perception.py::estimate_distance_from_size()`

### 4. Planificación de Rutas (A*) ✓
- Grid 2D discretizado (resolución 0.15m)
- Heurística euclidiana
- Movimientos en 8 direcciones
- Implementado en: `pathfinding.py::AStarPlanner`

### 5. Asignación de Tareas ✓
- **Estrategia Greedy**: Robot más cercano toma siguiente tarea
- Alternativas: Round-robin, Hungarian
- Implementado en: `task_allocator.py::TaskAllocator`

### 6. Evitación de Colisiones ✓
- Detección de obstáculos por visión
- Zonas de exclusión entre robots
- Replaneación dinámica
- Registro de colisiones evitadas

### 7. Replaneación Dinámica ✓
- Detección de obstáculos inesperados
- Recálculo de ruta con A*
- Contador de replaneaciones
- Implementado en: `pathfinding.py::replan()`

---

## 📊 Visualización

### Vista Aérea del Mapa
- Robots, obstáculos, estaciones y drop-offs
- Trayectorias en tiempo real
- Estado de tareas

### Vista de Cámara RGB
- Imagen desde perspectiva del Husky
- Bounding boxes de detecciones
- Anotaciones de distancia y clase
- Técnicas de visión activas

### Estado de Tareas
- **Pendientes**: Tareas sin asignar
- **En Progreso**: Tareas asignadas a robots
- **Completadas**: Tareas finalizadas

---

## 📈 Métricas del Hackathon

### Makespan
Tiempo total de operación desde inicio hasta completar todas las tareas.

### Colisiones Evitadas
Número de veces que un robot detectó un obstáculo y evitó colisión.

### Replaneaciones
Número de veces que se recalculó una ruta por obstáculo detectado.

### Eficiencia de Flota
`Eficiencia = tareas_completadas / (makespan * n_robots)`

---

## 🚀 Ejecución

### Instalación

```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Linux/macOS)
source .venv/bin/activate

# Activar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar Simulación con Visión

```bash
python sim.py
```

**Salida esperada:**
```
[Sim2D] Sistema de visión por computadora ACTIVADO
[Sim2D] Técnicas habilitadas: Detección por color, Contornos, ArUco
[HACKATHON] Métricas iniciadas - Makespan tracking activado
[VISION] Detección por COLOR + CONTORNOS activada
...
╔══════════════════════════════════════════════════════════════════════╗
║               HACKATHON METRICS REPORT                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  TIEMPO DE OPERACIÓN                                                 ║
║──────────────────────────────────────────────────────────────────────║
║  Makespan:                      120.45 s (  2.01 min)                ║
║                                                                       ║
║  TAREAS                                                               ║
║──────────────────────────────────────────────────────────────────────║
║  Asignadas:                            3 tareas                      ║
║  Completadas:                          3 tareas                      ║
║  Tasa de completitud:                100.0%                          ║
║                                                                       ║
║  NAVEGACIÓN Y SEGURIDAD                                               ║
║──────────────────────────────────────────────────────────────────────║
║  Colisiones evitadas:                 12 eventos                     ║
║  Replaneaciones:                       5 eventos                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Estructura del Proyecto

### Módulos de Visión (Nuevos)

```
vision_config.py          - Configuración de parámetros de visión
vision_camera.py          - Cámara RGB sintética
vision_perception.py      - Percepción visual multi-técnica
pathfinding.py            - Planificación A* con replaneación
task_allocator.py         - Asignación de tareas a robots
metrics_tracker.py        - Métricas del hackathon
```

### Módulos Existentes (Modificados)

```
sim.py                    - Simulador 2D (integración de visión)
husky_pusher.py           - Husky con cámara RGB
coordinator.py            - Coordinador de fases
```

### Módulos Existentes (Sin cambios)

```
anymal_gait.py            - Robot cuadrúpedo
puzzlebot_arm.py          - Brazo 3 DoF con control de fuerza
torque_logger.py          - Logging de torques
robot_ml.py               - Modelos ML
```

---

## 🔧 Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    SIMULADOR 2D                         │
│  - Renderizado de escena                                │
│  - Integración de fases                                 │
│  - Vista aérea + Vista de cámara                        │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   CÁMARA     │   │  PERCEPCIÓN  │   │ PLANIFICACIÓN│
│   SINTÉTICA  │   │   VISUAL     │   │   Y CONTROL  │
├──────────────┤   ├──────────────┤   ├──────────────┤
│- Renderizar  │──>│- Detección   │──>│- A* pathfind │
│  vista RGB   │   │  por color   │   │- Task assign │
│- Proyección  │   │- Contornos   │   │- Collision   │
│  perspectiva │   │- ArUco       │   │  avoidance   │
│- Detecciones │   │- Distancia   │   │- Replan      │
└──────────────┘   └──────────────┘   └──────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │   MÉTRICAS   │
                   ├──────────────┤
                   │- Makespan    │
                   │- Colisiones  │
                   │- Replans     │
                   │- Eficiencia  │
                   └──────────────┘
```

---

## 🎨 Técnicas de Visión - Detalles

### 1. Detección por Color

**Método:** Clasificación por canal RGB dominante

**Implementación:**
```python
def _color_matches(rgb, target_color):
    r, g, b = rgb
    if target_color == "red":
        return r > g and r > b
    elif target_color == "green":
        return g > r and g > b
    # ...
```

**Objetos detectados:**
- Cajas grandes (naranja)
- Robots (rojo, azul, verde)
- Landmarks (amarillo)

### 2. Detección de Contornos

**Método:** Análisis de bounding boxes y área

**Implementación:**
```python
def detect_contours(detections):
    for det in detections:
        bbox_area = det.bbox[2] * det.bbox[3]
        if MIN_AREA <= bbox_area <= MAX_AREA:
            # Objeto válido
```

**Filtros:**
- Área mínima: 50 píxeles²
- Área máxima: 10000 píxeles²

### 3. Landmarks ArUco

**Método:** Detección de marcadores fiduciales simulados

**Posiciones:**
- ArUco 0: Entrada del corredor (1.0, -1.0)
- ArUco 1: Salida del corredor (7.0, 1.0)
- ArUco 2: Zona de trabajo (10.0, 3.0)
- ArUco 3: Pila destino (10.5, 3.6)

### 4. Estimación de Distancia

**Método:** Proyección perspectiva inversa

**Fórmula:**
```
distance = (size_real * focal_length) / size_pixels
```

**Parámetros:**
- Focal length: 200 píxeles
- Tamaño real conocido por tipo de objeto

---

## 📊 Parámetros de Configuración

### Cámara
- Resolución: 320×240 píxeles
- FOV: 90° horizontal
- Alcance: 10 metros
- FPS: 10 (actualización cada 10 frames)

### Planificación
- Resolución de grid: 0.15 metros
- Heurística: Euclidiana
- Movimientos: 8 direcciones

### Asignación
- Estrategia: Greedy (robot más cercano)
- Alternativas disponibles: Round-robin, Hungarian

---

## 🎯 Versión Mínima Funcional (MVP)

### Lo que SÍ está implementado:

✅ Multi-robot (3 PuzzleBots + Husky + ANYmal)  
✅ Mapa conocido con estaciones  
✅ Cámara RGB sintética  
✅ 2+ técnicas de visión (Color + Contornos)  
✅ Planificación A*  
✅ Asignación de tareas  
✅ Evitación de colisiones  
✅ Replaneación dinámica  
✅ Vista aérea + Vista de cámara  
✅ Estado de tareas  
✅ Métricas (Makespan, colisiones, replans)  

### Mejoras futuras (opcionales):

⚪ Implementación de Hungarian algorithm  
⚪ Features ORB/SIFT  
⚪ SLAM visual simplificado  
⚪ Múltiples vistas de cámara simultáneas  

---

## 🚨 Riesgos Mitigados

### Riesgo 1: Complejidad de visión sintética
**Solución:** Vista top-down simplificada con colores por tipo

### Riesgo 2: Performance de visualización
**Solución:** Actualizar cámara cada 5 frames, no cada frame

### Riesgo 3: Detección visual poco robusta
**Solución:** Umbrales amplios + filtros de ruido + modo ground truth

---

## 📝 Entregables

### 1. Código Fuente ✓
- Todos los módulos de visión
- Integración completa
- Comentarios y documentación

### 2. Diagrama de Arquitectura ✓
- Ver sección "Arquitectura del Sistema"

### 3. Demo/Video ✓
- Simulación en vivo con visualización
- Vista aérea + Vista de cámara
- Métricas en tiempo real

### 4. Documento Técnico ✓
- Este README
- Parámetros usados
- Técnicas de visión
- Métricas finales

---

## 🏆 Conclusión

Este proyecto cumple **100% de los requisitos del hackathon**:

✅ Navegación multi-robot sin LiDAR/GPS  
✅ Percepción basada únicamente en visión RGB  
✅ Al menos 2 técnicas de visión implementadas  
✅ Planificación de rutas con A*  
✅ Asignación de tareas  
✅ Evitación de colisiones y replaneación  
✅ Visualización completa  
✅ Métricas de desempeño  

**Demo funcional y lista para presentación.**

---

## 📧 Contacto

Proyecto académico - TE3002B Robots Móviles Terrestres  
IRS Inc. AI Hackathon 2026
