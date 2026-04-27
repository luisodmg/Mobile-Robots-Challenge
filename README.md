# Almacen Robotico Autonomo

Simulacion 2D de un sistema multi-robot para un almacen autonomo, desarrollada en Python con enfoque en coordinacion por fases, cinematica, control de fuerza y visualizacion de metricas.

El proyecto integra tres subsistemas principales:

- Husky A200 para despejar un corredor empujando cajas grandes.
- ANYmal cuadrupedo para transportar PuzzleBots a zona de trabajo.
- PuzzleBots con brazo 3 DoF para apilar cajas pequenas con control de fuerza.

Adicionalmente, incorpora un modulo XArm simplificado para descarga de PuzzleBots como extension del flujo principal.

## 1. Objetivo del proyecto

Implementar y demostrar un pipeline completo de manipulacion y logistica robotica en simulacion:

1. Despejar un corredor de obstaculos.
2. Transportar robots moviles sobre una plataforma cuadrupeda.
3. Ejecutar apilado colaborativo con restricciones de orden y sincronizacion.
4. Registrar y reportar torques durante el contacto en manipulacion.

## 2. Alcance funcional

### 2.1 Fase 1 - Husky despeja corredor

- Modelo de locomocion tipo skid-steer.
- Compensacion de deslizamiento en velocidad lineal y angular.
- Sensor LiDAR 2D simulado para deteccion de obstaculos.
- Estrategia de empuje de cajas hacia fuera del corredor.
- Implementacion bloqueante y no bloqueante (maquina de estados para animacion fluida).
- Secuencia continua de cajas sin regresar a la posicion de inicio entre empujes.

### 2.2 Fase 2 - ANYmal transporta PuzzleBots

- Marcha tipo trote con pares diagonales.
- FK/IK por pata.
- Monitoreo continuo de singularidades por determinante del Jacobiano, det(J).
- Navegacion al objetivo con desaceleracion cerca del destino.

### 2.3 Fase 2.5 - Descarga con XArm (extra)

- Dos brazos XArm simplificados.
- Alcance e IK basica para recoger y colocar PuzzleBots.
- Reubicacion de PuzzleBots a posiciones de trabajo.

### 2.4 Fase 3 - PuzzleBots apilan cajas

- Brazo planar de 3 DoF en cada PuzzleBot.
- Pick and place con trayectorias cartesianas.
- Control de fuerza por Jacobiano transpuesto:

  tau = J^T * f

- Sincronizacion por eventos para asegurar orden C -> B -> A.
- Manejo de zonas de exclusion para evitar interferencias.

### 2.5 Integracion de ML en la logica de los robots

El proyecto incorpora una capa ligera de machine learning para ajustar el comportamiento
de los robots sin reemplazar la logica clasica de control. La idea no es que el modelo
"mande" por completo, sino que actue como un asesor numerico encima de las reglas ya
existentes.

En la implementacion actual se usa una politica compartida en [robot_ml_policy.py](robot_ml_policy.py), basada en modelos de random forest entrenados con datos sinteticos generados a partir del comportamiento esperado del propio sistema. Esa politica entrega tres tipos de recomendaciones:

- Husky: escala la velocidad y el giro segun distancia al objetivo, error angular, distancia al obstaculo mas cercano y si el robot esta empujando una caja.
- ANYmal: ajusta la agresividad de la marcha segun distancia al destino, payload y el valor minimo de det(J) observado en las patas.
- PuzzleBot: modifica la fuerza de agarre y la suavidad de descenso segun margen de workspace, det(J) y altura del objetivo.

La logica general es la siguiente:

1. El robot calcula primero su comando clasico usando reglas conocidas, por ejemplo control proporcional, IK, o control de fuerza.
2. Antes de ejecutar ese comando, consulta la politica ML con las variables de contexto relevantes.
3. El resultado de la politica no sustituye al controlador, sino que escala o modula el comando final.
4. Si la politica indica una situacion menos segura, el sistema sigue funcionando con la logica base, solo que de forma mas conservadora.

Esto tiene dos ventajas principales:

- Permite introducir ML en el pipeline sin perder interpretabilidad ni estabilidad.
- Facilita una futura migracion a modelos mas avanzados, como random forest, usando las mismas variables de entrada.

En otras palabras, el ML aqui cumple el papel de capa adaptativa: aprende patrones de decision a partir de variables ya disponibles en la simulacion y ajusta el comportamiento de cada robot de forma local.

## 3. Estructura del repositorio

- anymal_gait.py
  - Cinematica y marcha de ANYmal.
  - Deteccion de singularidades por pata.
- coordinator.py
  - Maquina de estados global del reto.
  - Orquestacion de fases, metricas y transiciones.
- husky_pusher.py
  - Modelo Husky, LiDAR y estrategia de despeje.
- robot_ml_policy.py
  - Politica ML compartida para ajustar decisiones de Husky, ANYmal y PuzzleBot.
  - Capa adaptativa ligera basada en random forest.
- puzzlebot_arm.py
  - FK, IK, Jacobiano, fuerza->torque y pick/place.
- sim.py
  - Simulador 2D animado e integrador de las fases.
  - Exporta visualizaciones y activa reportes de torque.
- torque_logger.py
  - Logging, analisis estadistico y graficas de torque.
- requirements.txt
  - Dependencias de Python.

Archivos de salida generados en ejecucion:

- results/sim_output.png
- results/metrics.png
- results/torque_report.json
- results/torque_analysis.png

## 4. Requisitos

- Python 3.10 o superior (recomendado).
- Dependencias:
  - numpy >= 2.0.0
  - matplotlib >= 3.8.0

Notas:

- En algunos entornos, la backend grafica de matplotlib puede variar segun el sistema.
- El simulador intenta usar TkAgg cuando esta disponible.

## 5. Instalacion

Desde la raiz del proyecto:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Ejecucion

### 6.1 Simulacion completa con visualizacion y reportes

```bash
python sim.py
```

Esto ejecuta las tres fases, actualiza la animacion en vivo y genera:

- results/sim_output.png: composicion con frames clave.
- results/metrics.png: resumen grafico de trayectoria y controles.
- results/torque_report.json: log estructurado de eventos y torques.
- results/torque_analysis.png: analisis visual del comportamiento de torque.

Al finalizar, la ventana de simulacion permanece abierta 3 segundos y se cierra automaticamente.

### 6.2 Ejecucion del coordinador (flujo por estados)

```bash
python coordinator.py
```

Muestra transiciones de fase, estado de exito/fallo y metricas finales en consola.

### 6.3 Prueba del modulo Husky

```bash
python husky_pusher.py
```

Permite validar el despeje de corredor y la respuesta del controlador skid-steer.

### 6.4 Prueba del modulo ANYmal

```bash
python anymal_gait.py
```

Ejecuta transporte a destino y reporte de det(J) por pata.

### 6.5 Prueba del brazo PuzzleBot

```bash
python puzzlebot_arm.py
```

Ejecuta tests basicos de FK/IK, Jacobiano y agarre/colocacion.

## 7. Flujo operativo del sistema

### 7.1 Orquestacion general

El coordinador sigue la secuencia:

IDLE -> PHASE1_HUSKY -> PHASE2_ANYMAL -> XARM_UNLOAD -> PHASE3_PUZZLEBOTS -> DONE

Si una fase critica falla, puede transicionar a ERROR.

### 7.2 Criterios de exito por fase

- Fase 1:
  - Todas las cajas grandes fuera del corredor.
- Fase 2:
  - ANYmal alcanza zona objetivo dentro de tolerancia.
  - Monitoreo de singularidades registrado.
- Fase 3:
  - Cajas pequenas apiladas en orden C-B-A.
  - Altura final esperada de pila: 3 * 0.05 m = 0.15 m.
  - Control de fuerza aplicado durante contacto.

## 8. Aspectos tecnicos relevantes

### 8.1 Husky - control y percepcion

- Sensor LiDAR 2D ray casting contra AABB de cajas.
- Ley de navegacion acoplada:
  - Avance solo cuando error angular es bajo.
  - Rotacion en sitio cuando no esta orientado a meta.
- Compensacion de deslizamiento para comandos de velocidad.

### 8.2 ANYmal - marcha y singularidades

- Marcha trote con alternancia de pares diagonales.
- IK por pata en cada ciclo de control.
- Singularidad detectada por umbral de det(J):
  - Umbral minimo usado: 1e-3.
- Registro historico de det(J) para analisis posterior.

### 8.3 PuzzleBotArm - manipulacion y fuerza

- Cinematica directa e inversa para brazo 3 DoF.
- Jacobiano analitico 3x3.
- Transformacion fuerza a torque:

  tau = J^T * f

- Registro de:
  - Torques por articulacion.
  - Magnitud total.
  - Valor de det(J) durante contacto.

## 9. Salidas y artefactos

### 9.1 sim_output.png

Imagen compuesta de momentos clave de las tres fases:

- Estado del corredor.
- Trayectoria/estado de robots.
- Evolucion del apilado.

### 9.2 metrics.png

Dashboard de metricas:

- Trayectoria 2D de ANYmal.
- det(J) por pata en escala logaritmica.
- Velocidad lineal y angular comandada/medida en Husky.

### 9.3 torque_report.json

Reporte estructurado con:

- Resumen global de operaciones.
- Estadisticas de torque.
- Eventos de control de fuerza.
- Historial detallado por operacion.

### 9.4 torque_analysis.png

Analisis grafico de torque:

- Magnitud temporal.
- Torques por articulacion.
- Evolucion de det(J).
- Distribucion media y desviacion por junta.

## 10. Reproducibilidad

El proyecto fija semillas aleatorias en varios puntos de entrada para reducir variabilidad entre corridas.

Para resultados comparables:

1. Mantener versiones de dependencias definidas en requirements.txt.
2. Ejecutar siempre desde un entorno virtual limpio.
3. Evitar cambiar parametros de dt o limites cinematicos sin documentarlo.

## 11. Problemas comunes y solucion

### 11.1 No aparece ventana de simulacion

- Verificar backend grafica de matplotlib disponible en el sistema.
- En entornos sin GUI, ejecutar y revisar archivos de salida PNG/JSON.

### 11.2 El brazo reporta punto fuera de workspace

- Es normal si un objetivo queda fuera del alcance cinematico.
- Ajustar posiciones objetivo o longitudes del brazo para pruebas alternativas.

### 11.3 Advertencias de singularidad

- El sistema monitorea det(J) y reporta eventos.
- Revisar torque_analysis.png y metrics.png para identificar tramos criticos.

### 11.4 Diferencias pequenas entre ejecuciones

- Existen componentes con ruido simulado y dinamica discreta.
- Esperar variaciones leves en tiempos y trayectorias, no en la logica global.

## 12. Posibles mejoras futuras

- Integrar planificacion de trayectorias con avoidance mas robusto.
- Incorporar control dinamico completo para contacto (impedancia/admitancia).
- Parametrizar escenario por archivo de configuracion.
- Exportar logs de toda la simulacion en formato tabular para analisis offline.
- Agregar pruebas automatizadas para regresion de metricas clave.

## 13. Creditos

Proyecto academico orientado a practicas de robotica computacional y sistemas multi-robot con manipulacion colaborativa.
