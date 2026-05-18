"""
vision_camera.py — Cámara RGB sintética para percepción visual.

Simula una cámara monocular RGB montada en cada robot.
Genera imágenes sintéticas del entorno desde la perspectiva del robot.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import vision_config as vc


@dataclass
class Detection:
    """Detección visual de un objeto."""
    object_type: str           # Tipo: "box_large", "box_small", "robot", "landmark"
    position_2d: np.ndarray    # Posición en imagen (u, v) píxeles
    position_3d: np.ndarray    # Posición estimada en mundo (x, y, z) metros
    distance: float            # Distancia estimada en metros
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, w, h)
    confidence: float          # Confianza de detección [0, 1]
    color: Tuple[int, int, int]  # Color RGB detectado


class VisionCamera:
    """Cámara RGB sintética para un robot.
    
    Simula una cámara monocular que captura el entorno desde la perspectiva
    del robot. Genera detecciones visuales basadas en proyección geométrica.
    """
    
    def __init__(self, robot_id: str, camera_offset: np.ndarray = np.array([0.15, 0.0, 0.2])):
        """
        Args:
            robot_id: Identificador del robot (e.g., "husky", "puzzlebot_0")
            camera_offset: Offset de la cámara respecto al centro del robot [x, y, z] metros
        """
        self.robot_id = robot_id
        self.camera_offset = camera_offset
        
        # Parámetros de cámara
        self.width = vc.CAMERA_WIDTH
        self.height = vc.CAMERA_HEIGHT
        self.fov = np.radians(vc.CAMERA_FOV)
        self.range_max = vc.CAMERA_RANGE
        self.focal_length = vc.FOCAL_LENGTH_PIXELS
        
        # Estado de la cámara
        self.position = np.zeros(3)  # Posición en mundo
        self.orientation = 0.0       # Orientación (yaw) en radianes
        
        # Buffer de imagen (simulada)
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Detecciones actuales
        self.detections: List[Detection] = []
        
        # Estadísticas
        self.frame_count = 0
        self.total_detections = 0
        
    def update_pose(self, robot_pos: np.ndarray, robot_theta: float):
        """Actualiza la pose de la cámara basada en la pose del robot.
        
        Args:
            robot_pos: Posición del robot [x, y, z] o [x, y]
            robot_theta: Orientación del robot (yaw) en radianes
        """
        if len(robot_pos) == 2:
            robot_pos = np.array([robot_pos[0], robot_pos[1], 0.0])
        
        # Transformar offset de cámara al marco mundial
        R = np.array([
            [np.cos(robot_theta), -np.sin(robot_theta), 0],
            [np.sin(robot_theta),  np.cos(robot_theta), 0],
            [0, 0, 1]
        ])
        
        self.position = robot_pos + R @ self.camera_offset
        self.orientation = robot_theta
        
    def capture(self, objects: List[Dict]) -> List[Detection]:
        """Captura una imagen y detecta objetos visibles.
        
        Args:
            objects: Lista de objetos en el mundo. Cada objeto es un dict con:
                - 'type': str (tipo de objeto)
                - 'pos': np.ndarray (posición [x, y, z])
                - 'size': float (tamaño característico)
                - 'color': Tuple[int, int, int] (color RGB)
                
        Returns:
            Lista de detecciones visuales
        """
        self.detections = []
        self.frame_count += 1
        
        for obj in objects:
            detection = self._detect_object(obj)
            if detection is not None:
                self.detections.append(detection)
                self.total_detections += 1
        
        return self.detections
    
    def _detect_object(self, obj: Dict) -> Optional[Detection]:
        """Detecta un objeto individual si está en el campo de visión.
        
        Args:
            obj: Diccionario con información del objeto
            
        Returns:
            Detection si el objeto es visible, None si no
        """
        obj_pos = obj['pos']
        obj_type = obj['type']
        obj_size = obj.get('size', 0.3)
        obj_color = obj.get('color', (255, 255, 255))
        
        # Vector de cámara a objeto
        delta = obj_pos - self.position
        distance = np.linalg.norm(delta)
        
        # Verificar rango
        if distance > self.range_max or distance < 0.1:
            return None
        
        # Transformar a marco de cámara
        dx_world = delta[0]
        dy_world = delta[1]
        
        # Rotar al marco de cámara
        cos_theta = np.cos(self.orientation)
        sin_theta = np.sin(self.orientation)
        
        dx_cam = dx_world * cos_theta + dy_world * sin_theta
        dy_cam = -dx_world * sin_theta + dy_world * cos_theta
        
        # Verificar si está delante de la cámara
        if dx_cam < 0.1:
            return None
        
        # Calcular ángulo horizontal
        angle_h = np.arctan2(dy_cam, dx_cam)
        
        # Verificar si está en FOV
        if abs(angle_h) > self.fov / 2:
            return None
        
        # Proyectar a imagen (proyección perspectiva simplificada)
        # u = focal_length * (y_cam / x_cam) + width/2
        u = int(self.focal_length * (dy_cam / dx_cam) + self.width / 2)
        v = int(self.height / 2)  # Simplificado: todo a altura media
        
        # Verificar si está dentro de la imagen
        if u < 0 or u >= self.width or v < 0 or v >= self.height:
            return None
        
        # Estimar tamaño en píxeles (proyección perspectiva)
        pixel_size = int((obj_size * self.focal_length) / distance)
        pixel_size = max(5, min(pixel_size, 100))  # Limitar tamaño
        
        # Bounding box
        bbox_x = max(0, u - pixel_size // 2)
        bbox_y = max(0, v - pixel_size // 2)
        bbox_w = min(pixel_size, self.width - bbox_x)
        bbox_h = min(pixel_size, self.height - bbox_y)
        
        # Estimar distancia desde tamaño en píxeles (inverso de proyección)
        estimated_distance = (obj_size * self.focal_length) / max(pixel_size, 1)
        
        # Agregar ruido a la estimación
        estimated_distance += np.random.normal(0, vc.DETECTION_NOISE_STD)
        
        # Confianza de detección (decrece con distancia)
        confidence = min(1.0, max(0.5, 1.0 - distance / self.range_max))
        
        # Probabilidad de detección
        if np.random.random() > vc.DETECTION_PROB:
            return None
        
        # Reconstruir posición 3D desde detección
        # Usando distancia estimada y ángulo
        pos_3d_cam = np.array([
            estimated_distance * np.cos(angle_h),
            estimated_distance * np.sin(angle_h),
            0.0
        ])
        
        # Transformar de vuelta a marco mundial
        R_inv = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0, 0, 1]
        ])
        pos_3d_world = self.position + R_inv @ pos_3d_cam
        
        return Detection(
            object_type=obj_type,
            position_2d=np.array([u, v]),
            position_3d=pos_3d_world,
            distance=estimated_distance,
            bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
            confidence=confidence,
            color=obj_color
        )
    
    def get_detections_by_type(self, object_type: str) -> List[Detection]:
        """Filtra detecciones por tipo de objeto.
        
        Args:
            object_type: Tipo de objeto a filtrar
            
        Returns:
            Lista de detecciones del tipo especificado
        """
        return [d for d in self.detections if d.object_type == object_type]
    
    def get_closest_detection(self, object_type: Optional[str] = None) -> Optional[Detection]:
        """Obtiene la detección más cercana.
        
        Args:
            object_type: Tipo de objeto (opcional, None = cualquier tipo)
            
        Returns:
            Detección más cercana o None si no hay detecciones
        """
        detections = self.detections if object_type is None else self.get_detections_by_type(object_type)
        
        if not detections:
            return None
        
        return min(detections, key=lambda d: d.distance)
    
    def render_debug_view(self) -> np.ndarray:
        """Renderiza una vista de debug de la cámara con anotaciones.
        
        Returns:
            Imagen RGB con detecciones anotadas
        """
        # Crear imagen base (fondo oscuro)
        img = np.full((self.height, self.width, 3), vc.COLORS["background"], dtype=np.uint8)
        
        # Dibujar detecciones
        for det in self.detections:
            x, y, w, h = det.bbox
            
            # Dibujar bounding box (simplificado como rectángulo de color)
            color = det.color
            img[y:y+h, x:x+w] = color
            
            # Dibujar borde
            border_thickness = 2
            img[y:y+border_thickness, x:x+w] = (255, 255, 255)  # Top
            img[y+h-border_thickness:y+h, x:x+w] = (255, 255, 255)  # Bottom
            img[y:y+h, x:x+border_thickness] = (255, 255, 255)  # Left
            img[y:y+h, x+w-border_thickness:x+w] = (255, 255, 255)  # Right
        
        return img
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de la cámara.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            "robot_id": self.robot_id,
            "frame_count": self.frame_count,
            "total_detections": self.total_detections,
            "current_detections": len(self.detections),
            "avg_detections_per_frame": self.total_detections / max(1, self.frame_count)
        }


# ---------------------------------------------------------------------------
# Funciones auxiliares para crear objetos visibles
# ---------------------------------------------------------------------------

def create_box_object(box_id: str, pos: np.ndarray, box_type: str = "large") -> Dict:
    """Crea un objeto caja para detección visual.
    
    Args:
        box_id: Identificador de la caja
        pos: Posición [x, y] o [x, y, z]
        box_type: "large" o "small"
        
    Returns:
        Diccionario de objeto
    """
    if len(pos) == 2:
        pos = np.array([pos[0], pos[1], 0.02])
    
    size = 0.4 if box_type == "large" else 0.1
    color = vc.COLORS["box_large"] if box_type == "large" else vc.COLORS["box_small_a"]
    
    return {
        "type": f"box_{box_type}",
        "id": box_id,
        "pos": pos,
        "size": size,
        "color": color
    }


def create_robot_object(robot_id: str, pos: np.ndarray, robot_type: str = "puzzlebot") -> Dict:
    """Crea un objeto robot para detección visual.
    
    Args:
        robot_id: Identificador del robot
        pos: Posición [x, y] o [x, y, z]
        robot_type: "husky", "anymal", "puzzlebot"
        
    Returns:
        Diccionario de objeto
    """
    if len(pos) == 2:
        pos = np.array([pos[0], pos[1], 0.0])
    
    color = vc.COLORS.get(robot_type, (255, 255, 255))
    
    return {
        "type": "robot",
        "id": robot_id,
        "pos": pos,
        "size": 0.35,
        "color": color
    }


def create_landmark_object(landmark_id: int, pos: np.ndarray) -> Dict:
    """Crea un landmark ArUco para detección visual.
    
    Args:
        landmark_id: ID del marcador ArUco
        pos: Posición [x, y, z]
        
    Returns:
        Diccionario de objeto
    """
    return {
        "type": "landmark",
        "id": f"aruco_{landmark_id}",
        "pos": pos,
        "size": vc.ARUCO_MARKER_SIZE,
        "color": vc.COLORS["landmark"]
    }
