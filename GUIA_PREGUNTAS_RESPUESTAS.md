# Guía Completa de Preguntas y Respuestas - Challenge Robots Móviles

## Índice
1. [Preguntas Generales del Proyecto](#preguntas-generales)
2. [Husky A200](#husky-a200)
3. [ANYmal Cuadrúpedo](#anymal-cuadrúpedo)
4. [PuzzleBots](#puzzlebots)
5. [Coordinador](#coordinador)
6. [Machine Learning](#machine-learning)
7. [Cinemática y Control](#cinemática-y-control)
8. [Implementación Técnica](#implementación-técnica)

---

## Preguntas Generales del Proyecto

### **P1: ¿Cuál es el objetivo principal del proyecto?**
El objetivo es implementar un pipeline completo de manipulación y logística robotica en simulación 2D que incluye:
1. Despejar un corredor de obstáculos con Husky
2. Transportar robots móviles (PuzzleBots) con ANYmal
3. Ejecutar apilado colaborativo con PuzzleBots
4. Registrar torques durante el contacto en manipulación

### **P2: ¿Qué robots participan en el sistema y qué función tiene cada uno?**
- **Husky A200**: Robot móvil skid-steer que despeja el corredor empujando cajas grandes
- **ANYmal**: Robot cuadrúpedo que transporta 3 PuzzleBots desde zona de inicio a zona de trabajo
- **PuzzleBots**: 3 robots móviles con brazo 3 DoF que apilan cajas pequeñas en orden C→B→A
- **XArm** (opcional): Brazos manipuladores que descargan PuzzleBots del ANYmal

### **P3: ¿Cuáles son las fases del sistema?**
1. **Fase 1**: Husky despeja corredor empujando cajas grandes
2. **Fase 2**: ANYmal transporta PuzzleBots a zona de trabajo
3. **Fase 2.5** (opcional): XArm descarga PuzzleBots
4. **Fase 3**: PuzzleBots apilan cajas pequeñas con control de fuerza

### **P4: ¿Qué archivos generan las salidas del sistema?**
- `results/sim_output.png`: Imagen compuesta con 6 frames clave de la simulación
- `results/metrics.png`: Dashboard con trayectorias, det(J), velocidades
- `results/torque_report.json`: Log estructurado de eventos y torques
- `results/torque_analysis.png`: Análisis visual del comportamiento de torque

### **P5: ¿Qué dependencias requiere el proyecto?**
- Python 3.10 o superior
- numpy >= 2.0.0
- matplotlib >= 3.8.0

---

## Husky A200

### **P6: ¿Qué tipo de locomoción tiene el Husky?**
Tiene locomoción **skid-steer** (diferencial), donde dos ruedas a cada lado giran a la misma velocidad. El control se realiza mediante velocidad lineal (v) y velocidad angular (ω).

### **P7: ¿Cómo funciona el controlador skid-steer con compensación de deslizamiento?**
El controlador aplica una compensación por deslizamiento:
- `v_meas = v_cmd * (1 - slip_factor)` donde slip_factor = 0.05
- `w_meas = w_cmd * (1 - slip_factor * 0.5)`
- Luego compensa los comandos: `v_comp = v_cmd / (1 - slip_factor)`

Esto modela que en robots skid-steer reales, las ruedas patinan y la velocidad real es menor a la comandada.

### **P8: ¿Qué es el LiDAR 2D simulado y cómo funciona?**
Es un sensor que emite 180 rayos en un arco de 180° (-90° a +90°). Cada rayo calcula la distancia al obstáculo más cercano usando **ray casting contra AABB** (Axis-Aligned Bounding Box). Implementa el algoritmo de intersección rayo-caja para detectar cajas en el corredor.

### **P9: ¿Qué estrategia usa el Husky para despejar cajas?**
Usa una máquina de estados con 4 estados:
1. **IDLE**: Busca siguiente caja no despejada
2. **PRE_POSITIONING**: Va a un punto seguro antes de la caja
3. **POSITIONING**: Se posiciona detrás de la caja
4. **PUSHING**: Empuja la caja fuera del corredor

Para cada caja, el Husky se posiciona detrás y la empuja perpendicularmente hacia afuera del corredor.

### **P10: ¿Qué es la ley de navegación acoplada del Husky?**
Es una estrategia donde:
- Solo avanza cuando el error angular es bajo (< 0.5 rad)
- Si el error angular es alto, solo rota en sitio (v = 0)
- Esto evita que el robot se desvíe mientras intenta alinearse

### **P11: ¿Cómo se integra ML en el Husky?**
Usa **Logistic Regression** para clasificar si una maniobra es segura. Features:
- `min_range`: distancia mínima del LiDAR
- `velocity`: velocidad actual
- `angle_to_box`: ángulo hacia la caja

Output: `(is_safe: bool, confidence: float)` usando función sigmoide.

---

## ANYmal Cuadrúpedo

### **P12: ¿Qué tipo de marcha usa el ANYmal?**
Usa marcha **trote** (trot gait) donde pares diagonales se mueven juntos:
- Pares: (LF, RH) y (RF, LH)
- Fase de vuelo (swing): el pie se mueve hacia adelante
- Fase de apoyo (stance): el pie se mueve hacia atrás relativo al cuerpo

### **P13: ¿Qué es FK e IK en el contexto del ANYmal?**
- **FK (Forward Kinematics)**: Dados los ángulos de las articulaciones (q), calcula la posición del pie en coordenadas mundiales. Usa rotación del cuerpo + offsets de cadera + cinemática de la pata.
- **IK (Inverse Kinematics)**: Dada la posición deseada del pie, calcula los ángulos de las articulaciones necesarios. Usa ley de cosenos y geometría trigonométrica.

### **P14: ¿Cómo se detectan singularidades en el ANYmal?**
Se monitorea el **determinante del Jacobiano** (det(J)) para cada pata:
- Si `det(J) < 1e-3`, se considera una singularidad
- Se limita `cos_kfe` a 0.995 en vez de 1.0 para evitar que la rodilla se estire completamente (evita singularidad de frontera)
- Los eventos de singularidad se registran en `singularity_events`

### **P15: ¿Qué es el Jacobiano y por qué es importante?**
El Jacobiano J es una matriz 3×3 que relaciona velocidades articulares con velocidades del end-effector:
- `v_tip = J * q_dot`
- El determinante indica si la configuración es singular (det ≈ 0)
- Se usa para control de fuerza: `τ = J^T * f`

### **P16: ¿Cómo se integra ML en el ANYmal?**
Usa **Linear Regression** para predecir el tiempo de llegada (ETA):
- Features: `[distance, payload_kg]`
- Modelo: `time = 3.0*distance + 0.5*payload + 5.0`
- Output: tiempo estimado en segundos

### **P17: ¿Cuál es el payload del ANYmal y por qué?**
El payload es **6 kg**, que representa el peso de los 3 PuzzleBots transportados (aprox. 2 kg cada uno). Este valor se usa en el predictor ML de tiempo.

---

## PuzzleBots

### **P18: ¿Qué es un PuzzleBot?**
Es un robot móvil diferencial con un brazo planar de **3 grados de libertad**:
- **q1**: rotación de la base (yaw, en plano XY)
- **q2**: ángulo del primer eslabón (hombro, plano vertical)
- **q3**: ángulo del segundo eslabón (codo, plano vertical)

### **P19: ¿Cómo funciona la cinemática directa del brazo PuzzleBot?**
Dado q = [q1, q2, q3]:
1. Calcula el radio horizontal: `r = l2*cos(q2) + l3*cos(q2+q3)`
2. Proyecta al plano XY: `x = r*cos(q1)`, `y = r*sin(q1)`
3. Calcula altura: `z = l1 + l2*sin(q2) + l3*sin(q2+q3)`

### **P20: ¿Cómo funciona la cinemática inversa del brazo PuzzleBot?**
Dada posición deseada p = [x, y, z]:
1. **q1** = `arctan2(y, x)` (ángulo de la base)
2. Calcula radio `r = sqrt(x² + y²)` y altura relativa `z' = z - l1`
3. Verifica alcanzabilidad: `|l2 - l3| < D < l2 + l3` donde `D = sqrt(r² + z'²)`
4. **q3** = `-arccos((D² - l2² - l3²) / (2*l2*l3))` (ley de cosenos)
5. **q2** = `beta ± gamma` donde `beta = arctan2(z', r)` y `gamma` se calcula geométricamente

### **P21: ¿Qué es el control de fuerza y cómo se implementa?**
El control de fuerza usa el **Jacobiano transpuesto**:
- Fórmula: `τ = J^T * f`
- Donde τ son los torques articulares, J es el Jacobiano, f es la fuerza deseada en el end-effector
- Se aplica durante el contacto al colocar cajas para garantizar un contacto suave

### **P22: ¿Por qué es importante el control de fuerza en el apilado?**
Permite que el brazo responda adecuadamente durante el contacto con la caja:
- Evita fuerzas excesivas que podrían dañar el robot o la caja
- Permite un contacto suave y controlado
- Es fundamental para manipulación colaborativa segura

### **P23: ¿Qué es la sincronización por eventos en Fase 3?**
Es un mecanismo para asegurar el orden de apilado C→B→A:
- `event_flags = {"C_completed": False, "B_completed": False, "A_completed": False}`
- El PuzzleBot B espera a que C_complete antes de empezar
- El PuzzleBot A espera a que B_complete antes de empezar
- Cuando un PuzzleBot termina, activa su evento correspondiente

### **P24: ¿Cómo se integra ML en los PuzzleBots?**
Usa **K-Means** (simplificado como 1-NN) para identificar zonas:
- Zonas predefinidas: pickup, stack, wait
- Dada una posición, encuentra la zona más cercana por distancia euclidiana
- Output: nombre de la zona donde está el PuzzleBot

---

## Coordinador

### **P25: ¿Qué es el coordinador y cuál es su función?**
Es una **máquina de estados** que orquesta las tres fases del reto:
- Transiciona entre fases: IDLE → PHASE1 → PHASE2 → XARM → PHASE3 → DONE
- Maneja errores y puede transicionar a ERROR
- Registra métricas de cada fase
- Coordina la sincronización entre robots

### **P26: ¿Cuáles son los estados del coordinador?**
```python
IDLE → PHASE1_HUSKY → PHASE2_ANYMAL → XARM_UNLOAD → PHASE3_PUZZLEBOTS → DONE
                                                              ↓
                                                           ERROR
```

### **P27: ¿Qué métricas registra el coordinador?**
- Tiempo de cada fase
- Éxito/fallo de cada fase
- Error final de posición del ANYmal
- Altura final de la pila
- Orden correcto de apilado (C-B-A)
- Tiempo total de la misión

### **P28: ¿Cómo se integra ML en el coordinador?**
Usa **Ridge Regression** para predecir el tiempo total de misión:
- Features: `[phase1_time, phase2_time, phase3_time]`
- Modelo: `total = 1.1*t1 + 1.05*t2 + 1.15*t3 + 10.0`
- El bias de 10.0 representa overhead de transiciones entre fases

### **P29: ¿Qué es XArm y cuándo se usa?**
XArm es un brazo manipulador de 6 DoF (simplificado) que:
- Recoge PuzzleBots del dorso del ANYmal
- Los coloca en posiciones de trabajo
- Es una fase opcional (puntos extra)
- Usa IK simplificada con ley de cosenos

---

## Machine Learning

### **P30: ¿Qué modelos ML se usan en el proyecto?**
1. **Husky**: Logistic Regression (clasificación binaria de seguridad)
2. **ANYmal**: Linear Regression (regresión para tiempo de llegada)
3. **PuzzleBot**: K-Means/1-NN (clasificación de zonas)
4. **Coordinator**: Ridge Regression (regresión para tiempo total)

### **P31: ¿Por qué se eligieron estos modelos específicos?**
- **Simplicidad**: Son modelos lineales o casi lineales, fáciles de interpretar
- **Eficiencia**: No requieren entrenamiento en tiempo de ejecución
- **Interpretabilidad**: Los pesos tienen significado físico claro
- **Adecuación**: Cada modelo resuelve un problema específico apropiado para su tipo

### **P32: ¿Los modelos ML están entrenados con datos reales?**
No, los modelos usan **pesos pre-configurados** basados en conocimiento del dominio:
- HuskyML: pesos heurísticos para seguridad
- ANYmalML: pesos basados en física (distancia/velocidad)
- PuzzleBotML: zonas predefinidas manualmente
- CoordinatorML: factores basados en overhead esperado

### **P33: ¿Dónde se ejecutan las predicciones ML en el flujo de simulación?**
- **Husky**: Durante Fase 1, al evaluar seguridad de maniobras (línea ~315 en sim.py)
- **ANYmal**: Al inicio de Fase 2, para predecir ETA (línea ~387 en sim.py)
- **PuzzleBot**: Durante Fase 3, al identificar zonas (línea ~490 en sim.py)
- **Coordinator**: Al final, para predecir tiempo total (línea ~575 en sim.py)

### **P34: ¿Cómo se podría mejorar la integración ML?**
1. Entrenar modelos con datos reales de simulaciones
2. Usar modelos más complejos (redes neuronales) si hay suficientes datos
3. Implementar aprendizaje online (actualizar pesos durante ejecución)
4. Agregar más features relevantes (historial, contexto, etc.)

---

## Cinemática y Control

### **P35: ¿Qué es la cinemática directa?**
Es el cálculo de la posición del end-effector dado los ángulos de las articulaciones. Para el brazo PuzzleBot:
- Input: q = [q1, q2, q3]
- Output: p = [x, y, z]
- Usa trigonometría para propagar transformaciones desde la base hasta la punta

### **P36: ¿Qué es la cinemática inversa?**
Es el cálculo de los ángulos articulares necesarios para alcanzar una posición deseada del end-effector. Para el brazo PuzzleBot:
- Input: p = [x, y, z]
- Output: q = [q1, q2, q3]
- Usa ley de cosenos y geometría inversa
- Puede tener múltiples soluciones o ninguna (fuera del workspace)

### **P37: ¿Qué es el workspace de un robot?**
Es el conjunto de todas las posiciones que el end-effector puede alcanzar. Para el PuzzleBot:
- Radio máximo: l2 + l3 = 0.12 + 0.10 = 0.22 m
- Radio mínimo: |l2 - l3| = 0.02 m
- Altura: depende de l1 y la configuración
- Si un punto está fuera, el IK retorna None

### **P38: ¿Qué es una singularidad cinemática?**
Es una configuración donde el Jacobiano pierde rango (det(J) ≈ 0), lo que significa:
- El robot no puede moverse en ciertas direcciones
- Las velocidades articulares se vuelven infinitas para movimientos finitos
- El control se vuelve inestable
- En el PuzzleBot, ocurre cuando el brazo está completamente extendido

### **P39: ¿Qué es el Jacobiano transpuesto y para qué sirve?**
El Jacobiano transpuesto J^T se usa en control de fuerza:
- `τ = J^T * f` transforma fuerzas cartesianas a torques articulares
- Es el método más simple de control de fuerza (no requiere inversión de J)
- Funciona bien para tareas de contacto donde se desea aplicar una fuerza específica

### **P40: ¿Qué es la compensación de deslizamiento?**
Es una técnica para modelar y corregir el patinaje en robots con tracción diferencial:
- Las ruedas patinan, así que v_real < v_cmd
- Se modela con un factor: v_real = v_cmd * (1 - slip)
- El controlador compensa: v_cmd_compensado = v_cmd / (1 - slip)
- En el Husky, slip_factor = 0.05 (5% de deslizamiento)

---

## Implementación Técnica

### **P41: ¿Qué estructura de archivos tiene el proyecto?**
- `sim.py`: Simulador 2D principal con visualización animada
- `coordinator.py`: Máquina de estados global
- `husky_pusher.py`: Modelo Husky, LiDAR y estrategia de despeje
- `anymal_gait.py`: Cinemática y marcha de ANYmal
- `puzzlebot_arm.py`: FK, IK, Jacobiano y pick/place del brazo
- `torque_logger.py`: Logging y análisis de torques
- `robot_ml.py`: Modelos de Machine Learning
- `env_config.py`: Configuración del entorno

### **P42: ¿Cómo se ejecuta la simulación completa?**
```bash
python sim.py
```
Esto ejecuta las tres fases con visualización en vivo y genera los archivos de salida en `results/`.

### **P43: ¿Qué es el modo no-bloqueante y por qué se usa?**
El modo no-bloqueante permite animaciones fluidas:
- En lugar de esperar a que una tarea termine, retorna False cada paso
- Una máquina de estados rastrea el progreso
- Permite que el simulador actualice la visualización entre pasos
- Se usa en Husky (clear_corridor_step) y PuzzleBots (pick_and_stack_nonblocking)

### **P44: ¿Cómo se registran los torques durante el contacto?**
Se usa el módulo `torque_logger.py`:
- `log_torque_data()`: registra torques por articulación
- `log_force_control_event()`: registra eventos de control de fuerza
- Genera `torque_report.json` con estructura detallada
- Genera `torque_analysis.png` con visualizaciones

### **P45: ¿Qué información contiene el torque_report.json?**
- Resumen global de operaciones
- Estadísticas de torque (media, desviación, máximo)
- Eventos de control de fuerza con fórmula usada
- Historial detallado por operación con timestamps
- Valores de det(J) durante contacto

### **P46: ¿Cómo se implementa la evitación de obstáculos?**
En los PuzzleBots, se usa un enfoque de campos de potencial:
- Dirección deseada hacia el objetivo
- Si cerca de un obstáculo, agrega:
  - Componente perpendicular para esquivar
  - Componente de repulsión para alejarse
- Normaliza la dirección resultante

### **P47: ¿Qué es ray casting y cómo se usa en el LiDAR?**
Ray casting es el algoritmo para calcular intersección de un rayo con objetos:
- Para cada ángulo del LiDAR, se define un rayo
- Se calcula la intersección con cada caja (AABB)
- Se usa el algoritmo de Liang-Barsky o similar para intersección rayo-rectángulo
- Retorna la distancia mínima a cualquier caja

### **P48: ¿Cómo se asegura el orden de apilado C-B-A?**
Mediante **sincronización por eventos**:
- Cada PuzzleBot tiene una caja asignada: PB0→C, PB1→B, PB2→A
- PB1 espera a que evento "C_completed" sea True
- PB2 espera a que evento "B_completed" sea True
- Cuando un PuzzleBot termina, activa su evento
- Esto asegura el orden sin necesidad de scheduling centralizado

### **P49: ¿Qué son las zonas de exclusión?**
Son áreas alrededor de otros robots donde un PuzzleBot no puede entrar:
- Evita colisiones entre PuzzleBots
- Se define como (centro, radio)
- Un PuzzleBot verifica si su objetivo está en alguna zona de exclusión
- Si está, espera o busca ruta alternativa

### **P50: ¿Cómo se visualiza la simulación?**
Usando matplotlib con animación:
- Figura con 2 subplots: escena principal y panel de información
- Función `_refresh_live_view()` redibuja cada frame
- Colores definidos en diccionario COLORS
- Parches de matplotlib para robots, cajas, zonas
- `plt.pause(0.001)` para actualización fluida

### **P51: ¿Qué es el dt (delta time) y por qué es importante?**
dt es el paso de tiempo de simulación (generalmente 0.05s):
- Determina la resolución temporal de la simulación
- Afecta la precisión de integración cinemática
- Más pequeño = más preciso pero más lento
- Debe ser consistente entre todos los módulos

### **P52: ¿Cómo se manejan los errores en el sistema?**
- El coordinador puede transicionar a estado ERROR
- Si una fase crítica falla, aborta la misión
- Fase 2 puede continuar aunque el ANYmal no llegue exactamente
- Se registran errores en métricas para análisis posterior

### **P53: ¿Qué es la ley de cosenos y dónde se usa?**
La ley de cosenos: `c² = a² + b² - 2ab*cos(C)`
Se usa en IK para calcular q3 (ángulo del codo):
- Dados l2, l3 (longitudes de eslabones) y D (distancia al objetivo)
- `cos(q3) = (D² - l2² - l3²) / (2*l2*l3)`
- `q3 = arccos(cos(q3))`

### **P54: ¿Por qué se limita cos_kfe a 0.995 en ANYmal?**
Para evitar singularidad de frontera:
- Si cos_kfe = 1.0, la rodilla está completamente extendida (q3 = 0)
- Esto causa det(J) ≈ 0 (singularidad)
- Limitando a 0.995, se fuerza que la rodilla siempre esté ligeramente flexionada
- Esto garantiza det(J) > 1e-3 (configuración segura)

### **P55: ¿Cómo se integran los modelos ML en sim.py?**
```python
from robot_ml import ml_system

# Husky
is_safe, conf = ml_system.husky_check_safety(min_range, velocity, angle)

# ANYmal
eta = ml_system.anymal_predict_eta(distance, payload)

# PuzzleBot
zone = ml_system.puzzlebot_get_zone(position)

# Coordinator
total_time = ml_system.coordinator_predict_mission(t1, t2, t3)
```

### **P56: ¿Qué es el archivo requirements.txt?**
Contiene las dependencias del proyecto:
```
numpy>=2.0.0
matplotlib>=3.8.0
```
Se instala con: `pip install -r requirements.txt`

### **P57: ¿Cómo se reproducen los resultados entre ejecuciones?**
Fijando semillas aleatorias:
```python
import random
random.seed(42)
np.random.seed(42)
```
Esto reduce variabilidad en ruido simulado y dinámica discreta.

### **P58: ¿Qué problemas comunes pueden ocurrir?**
1. **No aparece ventana**: verificar backend de matplotlib
2. **Brazo reporta punto fuera de workspace**: ajustar posiciones objetivo
3. **Advertencias de singularidad**: revisar torque_analysis.png
4. **Diferencias entre ejecuciones**: esperado por ruido simulado

### **P59: ¿Cómo se puede extender el proyecto?**
- Integrar planificación de trayectorias con avoidance más robusto
- Incorporar control dinámico completo (impedancia/admitancia)
- Parametrizar escenario por archivo de configuración
- Exportar logs en formato tabular para análisis offline
- Agregar pruebas automatizadas para regresión

### **P60: ¿Qué representa el color de las patas del ANYmal en visualización?**
- **Verde**: det(J) > 1e-3 (configuración segura)
- **Naranja**: det(J) ≤ 1e-3 (singularidad detectada)
- Esto permite identificar visualmente tramos críticos de la marcha

---

## Preguntas de Diseño y Arquitectura

### **P61: ¿Por qué se usó una máquina de estados en lugar de un script secuencial?**
- Permite transiciones flexibles entre fases
- Facilita manejo de errores y recuperación
- Soporta modo no-bloqueante para animaciones
- Es más extensible (agregar nuevas fases es fácil)

### **P62: ¿Por qué se separó la lógica en múltiples archivos?**
- **Modularidad**: cada archivo tiene responsabilidad clara
- **Mantenibilidad**: easier to debug y modificar componentes individuales
- **Reutilización**: se pueden probar módulos independientemente
- **Escalabilidad**: agregar nuevos robots o fases es más simple

### **P63: ¿Por qué se usó matplotlib en lugar de un motor de juego?**
- **Simplicidad**: no requiere dependencias adicionales
- **Integración con Python**: nativo para visualización científica
- **Adecuado**: simulación 2D no requiere renderizado complejo
- **Portabilidad**: funciona en la mayoría de sistemas

### **P64: ¿Cuál es la diferencia entre modo bloqueante y no-bloqueante?**
- **Bloqueante**: función espera hasta completar la tarea (ej: `goto()`)
- **No-bloqueante**: función retorna False cada paso, máquina de estados rastrea progreso (ej: `clear_corridor_step()`)
- El no-bloqueante es necesario para animaciones fluidas

### **P65: ¿Por qué el control de fuerza usa Jacobiano transpuesto en lugar de dinámica completa?**
- **Simplicidad**: no requiere modelo dinámico del robot
- **Eficiencia**: J^T * f es O(n²) vs inversión de matriz dinámica O(n³)
- **Adecuado**: para tareas de contacto simples es suficiente
- **Robustez**: no requiere estimación de parámetros dinámicos

---

## Preguntas Avanzadas

### **P66: ¿Cómo se calcularía el Jacobiano analíticamente vs numéricamente?**
- **Analítico**: derivadas parciales de la FK (implementado en puzzlebot_arm.py)
- **Numérico**: diferencias finitas: `J[:,i] ≈ (FK(q+ε) - FK(q-ε)) / (2ε)`
- El analítico es más exacto y eficiente, el numérico es más fácil de implementar

### **P67: ¿Qué es el determinante del Jacobiano y qué indica?**
- det(J) es el volumen del paralelepípedo formado por las columnas de J
- det(J) = 0 indica singularidad (pérdida de un grado de libertad)
- det(J) grande indica buena manipulabilidad
- Se usa para monitorear calidad de configuración

### **P68: ¿Cómo se implementa la marcha trote matemáticamente?**
- Dos fases: swing (0.5T) y stance (0.5T)
- Pares diagonales desfasados por 0.5 en fase
- Trayectoria del pie: elíptica en swing, lineal en stance
- Offset local: `x = stride * t` en stance, `z = step_height * sin(π*t)` en swing

### **P69: ¿Qué es el problema de inversa cinemática redundante?**
Ocurre cuando el robot tiene más DoF que necesarios para la tarea (6 DoF para posición 3D + orientación 3D). El PuzzleBot no es redundante (3 DoF para posición 3D), pero si lo fuera, se necesitaría optimización para elegir entre múltiples soluciones.

### **P70: ¿Cómo se podría agregar dinámica al sistema?**
- Modelar masa e inercia de eslabones
- Usar ecuaciones de Euler-Lagrange o Newton-Euler
- Implementar control de torque en lugar de posición
- Agregar simulación de contacto con fuerzas reaccionales

---

## Resumen para Examen Oral

### Puntos clave que debes dominar:

1. **Arquitectura general**: 3 fases, 3 robots, coordinador
2. **Husky**: skid-steer, LiDAR, compensación deslizamiento, ley acoplada
3. **ANYmal**: trote, FK/IK, singularidades, det(J)
4. **PuzzleBot**: brazo 3 DoF, FK/IK, control de fuerza τ=J^T*f
5. **Coordinador**: máquina de estados, sincronización por eventos
6. **ML**: 4 modelos simples, integración no invasiva
7. **Cinemática**: FK, IK, Jacobiano, singularidades
8. **Control**: fuerza, compensación, evitación de obstáculos

### Fórmulas importantes:

- **Control de fuerza**: `τ = J^T * f`
- **Compensación deslizamiento**: `v_comp = v_cmd / (1 - slip)`
- **IK codo (ley de cosenos)**: `cos(q3) = (D² - l2² - l3²) / (2*l2*l3)`
- **Logistic Regression**: `P(y=1) = 1 / (1 + exp(-(w·x + b)))`
- **Linear Regression**: `y = w·x + b`

### Conceptos que debes explicar claramente:

1. Diferencia entre FK e IK
2. Qué es una singularidad y por qué es problemática
3. Cómo funciona la sincronización por eventos
4. Por qué se usa control de fuerza
5. Cómo se integra ML en el sistema
6. Qué hace cada fase del sistema

### Datos técnicos que memorizar:

- Husky: slip_factor = 0.05, 3 cajas grandes
- ANYmal: 4 patas, payload = 6 kg, det(J)_min = 1e-3
- PuzzleBot: 3 DoF, l1=0.05m, l2=0.12m, l3=0.10m
- Cajas pequeñas: altura = 0.05m, orden C-B-A
- dt típico: 0.05s

---

**Preparate explicando estos conceptos en voz alta y practicando la derivación de las fórmulas clave. ¡Buena suerte!**
