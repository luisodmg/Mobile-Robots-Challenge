"""
vision_config.py — Configuración de parámetros de visión por computadora.

Define colores, resoluciones, y parámetros de detección visual para el hackathon.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Parámetros de cámara
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 320          # píxeles
CAMERA_HEIGHT = 240         # píxeles
CAMERA_FOV = 90.0           # grados (campo de visión horizontal)
CAMERA_RANGE = 10.0         # metros (alcance máximo)
CAMERA_FPS = 10             # frames por segundo (actualización)

# ---------------------------------------------------------------------------
# Colores de objetos (RGB para detección por color)
# ---------------------------------------------------------------------------
COLORS = {
    # Robots
    "husky":       (231, 76, 60),    # Rojo
    "anymal":      (52, 152, 219),   # Azul
    "puzzlebot":   (46, 204, 113),   # Verde
    
    # Obstáculos y cajas
    "box_large":   (230, 126, 34),   # Naranja
    "box_small_a": (155, 89, 182),   # Púrpura
    "box_small_b": (26, 188, 156),   # Turquesa
    "box_small_c": (243, 156, 18),   # Amarillo
    
    # Zonas y landmarks
    "corridor":    (26, 35, 64),     # Azul oscuro
    "work_zone":   (26, 51, 32),     # Verde oscuro
    "pickup":      (39, 174, 96),    # Verde brillante
    "dropoff":     (192, 57, 43),    # Rojo oscuro
    "landmark":    (241, 196, 15),   # Amarillo brillante
    
    # Fondo
    "background":  (15, 17, 23),     # Negro azulado
    "floor":       (44, 62, 80),     # Gris azulado
}

# ---------------------------------------------------------------------------
# Umbrales de detección por color (HSV)
# ---------------------------------------------------------------------------
# Formato: (H_min, S_min, V_min, H_max, S_max, V_max)
COLOR_THRESHOLDS = {
    "red":     (0, 100, 100, 10, 255, 255),      # Rojo (obstáculos)
    "green":   (40, 50, 50, 80, 255, 255),       # Verde (pickups)
    "blue":    (100, 50, 50, 130, 255, 255),     # Azul (robots)
    "yellow":  (20, 100, 100, 30, 255, 255),     # Amarillo (landmarks)
    "orange":  (10, 100, 100, 20, 255, 255),     # Naranja (cajas grandes)
}

# ---------------------------------------------------------------------------
# Parámetros de landmarks ArUco simulados
# ---------------------------------------------------------------------------
ARUCO_DICT_SIZE = 4          # Número de marcadores ArUco
ARUCO_MARKER_SIZE = 0.15     # metros (tamaño físico del marcador)
ARUCO_POSITIONS = {
    0: np.array([1.0, -1.0, 0.5]),   # Entrada del corredor
    1: np.array([7.0, 1.0, 0.5]),    # Salida del corredor
    2: np.array([10.0, 3.0, 0.5]),   # Zona de trabajo
    3: np.array([10.5, 3.6, 0.5]),   # Pila destino
}

# ---------------------------------------------------------------------------
# Parámetros de detección de contornos
# ---------------------------------------------------------------------------
CONTOUR_MIN_AREA = 50        # píxeles² (área mínima para contorno válido)
CONTOUR_MAX_AREA = 10000     # píxeles² (área máxima)
CONTOUR_APPROX_EPSILON = 0.02  # Factor de aproximación de contorno

# ---------------------------------------------------------------------------
# Parámetros de estimación de distancia
# ---------------------------------------------------------------------------
# Calibración: distancia = (tamaño_real * focal_length) / tamaño_píxeles
FOCAL_LENGTH_PIXELS = 200.0  # Longitud focal en píxeles (calibrada)
KNOWN_OBJECT_SIZES = {
    "box_large": 0.4,        # metros (ancho de caja grande)
    "box_small": 0.1,        # metros (ancho de caja pequeña)
    "robot": 0.35,           # metros (ancho de robot)
    "aruco": 0.15,           # metros (tamaño de marcador ArUco)
}

# ---------------------------------------------------------------------------
# Parámetros de ruido y simulación
# ---------------------------------------------------------------------------
DETECTION_NOISE_STD = 0.02   # Desviación estándar del ruido en detección (metros)
DETECTION_PROB = 0.95        # Probabilidad de detección exitosa
OCCLUSION_ENABLED = True     # Habilitar oclusión de objetos

# ---------------------------------------------------------------------------
# Técnicas de visión habilitadas
# ---------------------------------------------------------------------------
VISION_TECHNIQUES = {
    "color_detection": True,      # Detección por color (Técnica 1)
    "contour_detection": True,    # Detección de contornos (Técnica 2)
    "aruco_landmarks": True,      # Landmarks ArUco (Técnica 3 - opcional)
    "distance_estimation": True,  # Estimación de distancia desde tamaño
}

# ---------------------------------------------------------------------------
# Configuración de visualización
# ---------------------------------------------------------------------------
SHOW_CAMERA_VIEW = True      # Mostrar vista de cámara en simulación
SHOW_DETECTIONS = True       # Mostrar bounding boxes de detecciones
SHOW_ANNOTATIONS = True      # Mostrar anotaciones de distancia/clase
CAMERA_UPDATE_RATE = 3       # Actualizar vista cada N frames (performance)
