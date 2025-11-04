#!/usr/bin/env python3
"""
Ejemplo de integración del detector de objetos con un sistema de navegación.

Este script muestra cómo:
1. Importar y usar la clase ObjectDetector
2. Procesar detecciones en tiempo real
3. Tomar decisiones de navegación basadas en la posición de objetos
"""

from yolo_detection_arm import ObjectDetector
import time
import sys


class NavigationController:
    """Controlador de navegación que usa detecciones de objetos"""
    
    def __init__(self, detector):
        """
        Inicializa el controlador de navegación
        
        Args:
            detector: Instancia de ObjectDetector
        """
        self.detector = detector
        self.target_detected = False
        self.target_position = None
        self.target_confidence = 0.0
        self.target_track_id = None
    
    def process_detections(self, detections):
        """
        Procesa detecciones y actualiza estado de navegación
        
        Args:
            detections: Lista de diccionarios con detecciones
            
        Returns:
            dict: Información de estado para navegación
        """
        if not detections:
            self.target_detected = False
            self.target_position = None
            return {
                'has_target': False,
                'action': 'ESPERAR',  # No hay objeto
                'confidence': 0.0,
                'position': None
            }
        
        # Usar la primera detección (la más confiable)
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        self.target_detected = True
        self.target_position = best_detection['position']
        self.target_confidence = best_detection['confidence']
        self.target_track_id = best_detection['track_id']
        
        # Determinar acción basada en posición
        if best_detection['position'] == "IZQUIERDA":
            action = "GIRAR_IZQUIERDA"
        elif best_detection['position'] == "CENTRO":
            action = "AVANZAR"
        else:  # DERECHA
            action = "GIRAR_DERECHA"
        
        return {
            'has_target': True,
            'action': action,
            'confidence': self.target_confidence,
            'position': self.target_position,
            'track_id': self.target_track_id
        }
    
    def execute_command(self, navigation_info):
        """
        Ejecuta comando de navegación (ejemplo)
        
        Args:
            navigation_info: Diccionario con información de navegación
        """
        action = navigation_info['action']
        
        if action == "ESPERAR":
            print("❌ Sin objetivo detectado - ESPERANDO...")
        elif action == "AVANZAR":
            print(f"➡️  AVANZAR (Objeto en CENTRO, confianza: {navigation_info['confidence']:.2f})")
        elif action == "GIRAR_IZQUIERDA":
            print(f"⬅️  GIRAR IZQUIERDA (Objeto en {navigation_info['position']}, confianza: {navigation_info['confidence']:.2f})")
        elif action == "GIRAR_DERECHA":
            print(f"➡️  GIRAR DERECHA (Objeto en {navigation_info['position']}, confianza: {navigation_info['confidence']:.2f})")


def example_standalone():
    """Ejemplo 1: Usar el detector en modo standalone (como antes)"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Modo Standalone (Visualización en tiempo real)")
    print("="*60)
    print("\nPara ejecutar: python yolo-detection-arm.py --camera 0 --model yolov8n.pt --conf 0.5")
    print("\nControles:")
    print("  - 'q' o ESC: Salir")
    print("  - ESPACIO: Pausar/Reanudar")


def example_integration():
    """Ejemplo 2: Integración con sistema de navegación"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Integración con Sistema de Navegación")
    print("="*60)
    
    try:
        # Crear detector con configuración optimizada para Raspberry Pi
        print("\n[INFO] Inicializando detector...")
        detector = ObjectDetector(
            model_path="yolov8n.pt",
            camera_idx=0,
            conf_threshold=0.5,
            imgsz=320,
            width=640,
            height=480,
            fps=30,
            tracker="bytetrack.yaml",
            half_precision=False,
            arm_optimize=True,  # Optimizaciones para ARM/Raspberry Pi
            verbose=True
        )
        
        # Crear controlador de navegación
        nav_controller = NavigationController(detector)
        
        print("\n[INFO] Detector listo. Procesando frames...")
        print("[INFO] Presiona Ctrl+C para salir\n")
        
        frame_count = 0
        start_time = time.time()
        
        # Loop principal de navegación
        try:
            while True:
                # Obtener detección de cámara
                result = detector.get_detection()
                frame_count += 1
                
                # Procesar detecciones
                nav_info = nav_controller.process_detections(result['objects'])
                
                # Ejecutar comando
                nav_controller.execute_command(nav_info)
                
                # Mostrar estadísticas cada 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"\n[STATS] Frame {frame_count} | FPS: {fps:.1f} | Tiempo: {elapsed:.1f}s")
                    
                # Pequeña pausa para no saturar CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupción detectada. Cerrando...")
        
        # Liberar recursos
        detector.close()
        
        # Mostrar estadísticas finales
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"\n[FINAL] Procesados {frame_count} frames en {elapsed:.1f}s")
        print(f"[FINAL] FPS promedio: {fps:.1f}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def example_callback_mode():
    """Ejemplo 3: Modo callback (para sistemas complejos)"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Modo Callback (para integración avanzada)")
    print("="*60)
    print("""
Este modo permite procesar detecciones en tiempo real con callbacks personalizados.

Uso:
    def on_object_detected(obj_info):
        # Tu lógica aquí
        print(f"Objeto detectado: {obj_info['position']}")
    
    detector = ObjectDetector(...)
    
    while True:
        result = detector.get_detection()
        for obj in result['objects']:
            on_object_detected(obj)
            
Ver navigation_integration_example.py para más detalles.
    """)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════╗
║   Ejemplos de Integración: Detector de Objetos YOLOv8 + ARM   ║
╚════════════════════════════════════════════════════════════════╝

Este script muestra 3 formas de usar el detector:
    """)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "standalone":
            example_standalone()
        elif sys.argv[1] == "integration":
            example_integration()
        elif sys.argv[1] == "callback":
            example_callback_mode()
        else:
            print(f"Opción desconocida: {sys.argv[1]}")
            print("Usa: python navigation_integration_example.py [standalone|integration|callback]")
    else:
        print("""
EJEMPLOS DE USO:

1. Modo Standalone (visualización):
   python yolo-detection-arm.py --camera 0 --model yolov8n.pt --arm-optimize --verbose
   
   Parámetros útiles:
   --camera 0        Índice de cámara (0 = primera cámara)
   --model           Modelo YOLOv8 (yolov8n.pt, yolov8s.pt, etc)
   --conf 0.5        Confianza mínima para detección
   --imgsz 320       Tamaño de imagen (menor = más rápido)
   --width 640       Ancho de captura
   --height 480      Alto de captura
   --arm-optimize    Activar optimizaciones para Raspberry Pi
   --verbose         Mostrar mensajes detallados
   --half            Usar precision FP16 (solo con GPU)

2. Integración con Navegación:
   python navigation_integration_example.py integration
   
   Esto ejecuta un ejemplo completo que:
   - Crea un detector optimizado para ARM
   - Procesa detecciones en tiempo real
   - Toma decisiones de navegación (AVANZAR, GIRAR, etc)
   - Muestra información de estado

3. Modo Callback (avanzado):
   python navigation_integration_example.py callback

IMPORTAR EN TU CÓDIGO:

    from yolo_detection_arm import ObjectDetector
    
    # Crear detector
    detector = ObjectDetector(
        model_path="yolov8n.pt",
        camera_idx=0,
        conf_threshold=0.5,
        arm_optimize=True,
        verbose=True
    )
    
    # Usar en loop
    while True:
        result = detector.get_detection()
        
        if result['detected']:
            for obj in result['objects']:
                print(f"Objeto detectado en: {obj['position']}")
                print(f"Confianza: {obj['confidence']:.2f}")
                print(f"Track ID: {obj['track_id']}")
        
        detector.close()

OUTPUT DEL DETECTOR:

El método get_detection() retorna:
    {
        'detected': bool,           # ¿Hay objetos detectados?
        'objects': [                # Lista de objetos
            {
                'class': 0,         # Clase (0 = persona)
                'position': str,    # 'IZQUIERDA', 'CENTRO', 'DERECHA'
                'confidence': float,# 0.0 a 1.0
                'track_id': int,    # ID de tracking (si disponible)
                'bbox': tuple,      # (x1, y1, x2, y2)
                'center': tuple     # (cx, cy)
            }
        ],
        'frame': np.ndarray,        # Frame capturado
        'fps': float,               # FPS actual
        'timestamp': float          # Timestamp del frame
    }

OPTIMIZACIONES PARA ARM/RASPBERRY PI:

--arm-optimize activa:
    - Configuración de threads optimizada (2 threads)
    - NEON Optimization (si disponible)
    - Detección automática de Raspberry Pi
    - Monitoreo de temperatura del CPU
    - Soporte para NCNN (compilación manual requerida)

Para máxima performance en Raspberry Pi:
    python yolo-detection-arm.py \\
        --camera 0 \\
        --model yolov8n.pt \\
        --imgsz 320 \\
        --width 320 \\
        --height 240 \\
        --arm-optimize \\
        --verbose
    """)
