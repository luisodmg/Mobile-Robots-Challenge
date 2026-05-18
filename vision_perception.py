"""
vision_perception.py — Módulo de percepción visual con múltiples técnicas.

Implementa las técnicas de visión requeridas por el hackathon:
1. Detección por color
2. Detección de contornos
3. Landmarks ArUco (opcional)
4. Estimación de distancia desde tamaño
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import vision_config as vc
from vision_camera import Detection


@dataclass
class VisualObstacle:
    """Obstáculo detectado visualmente."""
    position: np.ndarray    # Posición estimada [x, y]
    distance: float         # Distancia en metros
    angle: float            # Ángulo relativo en radianes
    size: float             # Tamaño estimado en metros
    confidence: float       # Confianza [0, 1]
    detection_method: str   # "color", "contour", "aruco"


@dataclass
class VisualLandmark:
    """Landmark visual (ArUco o feature)."""
    landmark_id: int        # ID del landmark
    position: np.ndarray    # Posición en mundo [x, y, z]
    distance: float         # Distancia en metros
    angle: float            # Ángulo relativo en radianes
    confidence: float       # Confianza [0, 1]


class VisionPerception:
    """Sistema de percepción visual multi-técnica.
    
    Procesa detecciones de cámara y extrae información útil para navegación.
    """
    
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        
        # Técnicas habilitadas
        self.use_color = vc.VISION_TECHNIQUES["color_detection"]
        self.use_contour = vc.VISION_TECHNIQUES["contour_detection"]
        self.use_aruco = vc.VISION_TECHNIQUES["aruco_landmarks"]
        
        # Estado
        self.obstacles: List[VisualObstacle] = []
        self.landmarks: List[VisualLandmark] = []
        
        # Estadísticas
        self.detections_by_method = {
            "color": 0,
            "contour": 0,
            "aruco": 0
        }
        
    def process_detections(self, detections: List[Detection], robot_theta: float) -> None:
        """Procesa detecciones de cámara y extrae obstáculos y landmarks.
        
        Args:
            detections: Lista de detecciones de la cámara
            robot_theta: Orientación actual del robot (radianes)
        """
        self.obstacles = []
        self.landmarks = []
        
        for det in detections:
            # Procesar según tipo de objeto
            if det.object_type.startswith("box"):
                obstacle = self._detection_to_obstacle(det, robot_theta, "color")
                if obstacle:
                    self.obstacles.append(obstacle)
                    self.detections_by_method["color"] += 1
                    
            elif det.object_type == "landmark":
                landmark = self._detection_to_landmark(det, robot_theta)
                if landmark:
                    self.landmarks.append(landmark)
                    self.detections_by_method["aruco"] += 1
                    
            elif det.object_type == "robot":
                # Otros robots también son obstáculos
                obstacle = self._detection_to_obstacle(det, robot_theta, "color")
                if obstacle:
                    self.obstacles.append(obstacle)
    
    def _detection_to_obstacle(self, det: Detection, robot_theta: float, method: str) -> Optional[VisualObstacle]:
        """Convierte una detección en un obstáculo visual.
        
        Args:
            det: Detección de cámara
            robot_theta: Orientación del robot
            method: Método de detección usado
            
        Returns:
            VisualObstacle o None
        """
        # Calcular ángulo relativo al robot
        dx = det.position_3d[0]
        dy = det.position_3d[1]
        angle_world = np.arctan2(dy, dx)
        angle_relative = self._normalize_angle(angle_world - robot_theta)
        
        # Estimar tamaño desde bounding box
        bbox_w = det.bbox[2]
        bbox_h = det.bbox[3]
        estimated_size = (det.distance * bbox_w) / vc.FOCAL_LENGTH_PIXELS
        
        return VisualObstacle(
            position=det.position_3d[:2],
            distance=det.distance,
            angle=angle_relative,
            size=estimated_size,
            confidence=det.confidence,
            detection_method=method
        )
    
    def _detection_to_landmark(self, det: Detection, robot_theta: float) -> Optional[VisualLandmark]:
        """Convierte una detección en un landmark visual.
        
        Args:
            det: Detección de cámara
            robot_theta: Orientación del robot
            
        Returns:
            VisualLandmark o None
        """
        # Extraer ID del landmark desde el nombre
        try:
            landmark_id = int(det.object_type.split("_")[-1]) if "_" in det.object_type else 0
        except:
            landmark_id = 0
        
        # Calcular ángulo relativo
        dx = det.position_3d[0]
        dy = det.position_3d[1]
        angle_world = np.arctan2(dy, dx)
        angle_relative = self._normalize_angle(angle_world - robot_theta)
        
        return VisualLandmark(
            landmark_id=landmark_id,
            position=det.position_3d,
            distance=det.distance,
            angle=angle_relative,
            confidence=det.confidence
        )
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza un ángulo al rango [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # -----------------------------------------------------------------------
    # Técnica 1: Detección por color
    # -----------------------------------------------------------------------
    
    def detect_by_color(self, detections: List[Detection], target_color: str) -> List[VisualObstacle]:
        """Detecta objetos por color específico.
        
        Args:
            detections: Lista de detecciones
            target_color: Color objetivo ("red", "green", "blue", etc.)
            
        Returns:
            Lista de obstáculos del color especificado
        """
        if not self.use_color:
            return []
        
        obstacles = []
        for det in detections:
            # Verificar si el color coincide (simplificado)
            if self._color_matches(det.color, target_color):
                obstacle = VisualObstacle(
                    position=det.position_3d[:2],
                    distance=det.distance,
                    angle=0.0,  # Se calculará después
                    size=0.3,
                    confidence=det.confidence,
                    detection_method="color"
                )
                obstacles.append(obstacle)
                self.detections_by_method["color"] += 1
        
        return obstacles
    
    @staticmethod
    def _color_matches(rgb: Tuple[int, int, int], target_color: str) -> bool:
        """Verifica si un color RGB coincide con un color objetivo.
        
        Args:
            rgb: Color RGB (r, g, b)
            target_color: Nombre del color objetivo
            
        Returns:
            True si coincide
        """
        r, g, b = rgb
        
        # Clasificación simple por canal dominante
        if target_color == "red":
            return r > g and r > b
        elif target_color == "green":
            return g > r and g > b
        elif target_color == "blue":
            return b > r and b > g
        elif target_color == "yellow":
            return r > 150 and g > 150 and b < 100
        elif target_color == "orange":
            return r > 200 and g > 100 and g < 200 and b < 100
        
        return False
    
    # -----------------------------------------------------------------------
    # Técnica 2: Detección de contornos
    # -----------------------------------------------------------------------
    
    def detect_contours(self, detections: List[Detection]) -> List[VisualObstacle]:
        """Detecta objetos por análisis de contornos.
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Lista de obstáculos detectados por contorno
        """
        if not self.use_contour:
            return []
        
        obstacles = []
        for det in detections:
            # Verificar área del bounding box
            bbox_area = det.bbox[2] * det.bbox[3]
            
            if vc.CONTOUR_MIN_AREA <= bbox_area <= vc.CONTOUR_MAX_AREA:
                obstacle = VisualObstacle(
                    position=det.position_3d[:2],
                    distance=det.distance,
                    angle=0.0,
                    size=np.sqrt(bbox_area) / vc.FOCAL_LENGTH_PIXELS * det.distance,
                    confidence=det.confidence * 0.9,  # Menor confianza que color
                    detection_method="contour"
                )
                obstacles.append(obstacle)
                self.detections_by_method["contour"] += 1
        
        return obstacles
    
    # -----------------------------------------------------------------------
    # Técnica 3: Landmarks ArUco
    # -----------------------------------------------------------------------
    
    def detect_aruco_landmarks(self, detections: List[Detection]) -> List[VisualLandmark]:
        """Detecta landmarks ArUco.
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Lista de landmarks detectados
        """
        if not self.use_aruco:
            return []
        
        landmarks = []
        for det in detections:
            if det.object_type == "landmark":
                try:
                    landmark_id = int(det.object_type.split("_")[-1]) if "_" in det.object_type else 0
                except:
                    landmark_id = 0
                
                landmark = VisualLandmark(
                    landmark_id=landmark_id,
                    position=det.position_3d,
                    distance=det.distance,
                    angle=0.0,
                    confidence=det.confidence
                )
                landmarks.append(landmark)
                self.detections_by_method["aruco"] += 1
        
        return landmarks
    
    # -----------------------------------------------------------------------
    # Estimación de distancia
    # -----------------------------------------------------------------------
    
    def estimate_distance_from_size(self, bbox_width: int, known_size: float) -> float:
        """Estima distancia desde el tamaño del objeto en la imagen.
        
        Args:
            bbox_width: Ancho del bounding box en píxeles
            known_size: Tamaño real conocido del objeto en metros
            
        Returns:
            Distancia estimada en metros
        """
        if bbox_width <= 0:
            return vc.CAMERA_RANGE
        
        distance = (known_size * vc.FOCAL_LENGTH_PIXELS) / bbox_width
        return min(distance, vc.CAMERA_RANGE)
    
    # -----------------------------------------------------------------------
    # Consultas útiles para navegación
    # -----------------------------------------------------------------------
    
    def get_obstacles_in_front(self, max_distance: float = 3.0, angle_threshold: float = np.pi/4) -> List[VisualObstacle]:
        """Obtiene obstáculos directamente enfrente del robot.
        
        Args:
            max_distance: Distancia máxima a considerar
            angle_threshold: Ángulo máximo desde el frente (radianes)
            
        Returns:
            Lista de obstáculos enfrente
        """
        return [
            obs for obs in self.obstacles
            if obs.distance <= max_distance and abs(obs.angle) <= angle_threshold
        ]
    
    def get_closest_obstacle(self) -> Optional[VisualObstacle]:
        """Obtiene el obstáculo más cercano."""
        if not self.obstacles:
            return None
        return min(self.obstacles, key=lambda obs: obs.distance)
    
    def get_min_range(self) -> float:
        """Obtiene la distancia al obstáculo más cercano (similar a LiDAR min range)."""
        closest = self.get_closest_obstacle()
        return closest.distance if closest else vc.CAMERA_RANGE
    
    def is_path_clear(self, distance: float = 1.0, angle_tolerance: float = np.pi/6) -> bool:
        """Verifica si el camino enfrente está despejado.
        
        Args:
            distance: Distancia a verificar
            angle_tolerance: Tolerancia angular
            
        Returns:
            True si el camino está despejado
        """
        obstacles_ahead = self.get_obstacles_in_front(distance, angle_tolerance)
        return len(obstacles_ahead) == 0
    
    def get_landmark_by_id(self, landmark_id: int) -> Optional[VisualLandmark]:
        """Obtiene un landmark específico por ID."""
        for lm in self.landmarks:
            if lm.landmark_id == landmark_id:
                return lm
        return None
    
    def get_statistics(self) -> Dict:
        """Obtiene estadísticas de percepción."""
        return {
            "robot_id": self.robot_id,
            "total_obstacles": len(self.obstacles),
            "total_landmarks": len(self.landmarks),
            "detections_by_method": self.detections_by_method.copy(),
            "techniques_enabled": {
                "color": self.use_color,
                "contour": self.use_contour,
                "aruco": self.use_aruco
            }
        }


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

def visualize_detections(detections: List[Detection], width: int = 320, height: int = 240) -> np.ndarray:
    """Visualiza detecciones en una imagen.
    
    Args:
        detections: Lista de detecciones
        width: Ancho de imagen
        height: Alto de imagen
        
    Returns:
        Imagen RGB con detecciones dibujadas
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for det in detections:
        x, y, w, h = det.bbox
        color = det.color
        
        # Dibujar bounding box
        img[y:y+h, x:x+w] = color
        
        # Dibujar borde blanco
        border = 2
        img[y:y+border, x:x+w] = (255, 255, 255)
        img[y+h-border:y+h, x:x+w] = (255, 255, 255)
        img[y:y+h, x:x+border] = (255, 255, 255)
        img[y:y+h, x+w-border:x+w] = (255, 255, 255)
    
    return img
