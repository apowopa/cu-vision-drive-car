#!/usr/bin/env python3
"""
Guía rápida de uso del detector de objetos con optimizaciones ARM
"""

# ============================================================================
# EJEMPLO 1: Modo Standalone - Visualización en tiempo real
# ============================================================================
# 
# Este es el modo que ya estaba disponible. Ejecuta:
#
#   cd /home/apowo/Projects/cu-vision-drive-car
#   python yolo-detection-arm.py --camera 0 --model yolov8n.pt --arm-optimize --verbose
#
# Controles:
#   - 'q' o ESC: Salir
#   - ESPACIO: Pausar/Reanudar
#
# Parámetros recomendados para Raspberry Pi:
#   python yolo-detection-arm.py \
#       --camera 0 \
#       --model yolov8n.pt \
#       --imgsz 320 \
#       --width 320 \
#       --height 240 \
#       --arm-optimize \
#       --verbose


# ============================================================================
# EJEMPLO 2: Integración con Script de Navegación
# ============================================================================
#
# Ejecuta el ejemplo de integración:
#
#   python navigation_integration_example.py integration
#
# Output esperado:
#   [INFO] Inicializando detector...
#   [INFO] Raspberry Pi detectada
#   [INFO] Optimizaciones ARM aplicadas
#   [INFO] Cargando modelo yolov8n.pt...
#   [INFO] Abriendo cámara 0...
#   [INFO] Captura: 640x480 @ 30.0FPS
#   [INFO] Detector listo. Procesando frames...
#   ➡️  AVANZAR (Objeto en CENTRO, confianza: 0.85)
#   ⬅️  GIRAR IZQUIERDA (Objeto en IZQUIERDA, confianza: 0.92)
#   [STATS] Frame 30 | FPS: 15.2 | Tiempo: 1.9s


# ============================================================================
# EJEMPLO 3: Usar como módulo en tu código
# ============================================================================
#
# En tu script de navegación principal, importa y usa:
#

from yolo_detection_arm import ObjectDetector

# Crear detector
detector = ObjectDetector(
    model_path="yolov8n.pt",      # Modelo
    camera_idx=0,                 # Cámara
    conf_threshold=0.5,           # Confianza mínima
    imgsz=320,                    # Tamaño de imagen
    width=640,                    # Ancho de captura
    height=480,                   # Alto de captura
    fps=30,                       # FPS
    tracker="bytetrack.yaml",     # Tracker
    half_precision=False,         # FP16 (solo GPU)
    arm_optimize=True,            # Optimizaciones ARM
    verbose=True                  # Mensajes detallados
)

# Loop de navegación
while True:
    # Obtener detección
    result = detector.get_detection()
    
    # Verificar si se detectó algo
    if result['detected']:
        print(f"FPS: {result['fps']:.1f} | Objetos detectados: {len(result['objects'])}")
        
        for obj in result['objects']:
            print(f"  - Posición: {obj['position']}")
            print(f"    Confianza: {obj['confidence']:.2f}")
            print(f"    Track ID: {obj['track_id']}")
            print(f"    Centro: {obj['center']}")
            
            # Tomar decisión de navegación
            if obj['position'] == 'IZQUIERDA':
                print("    → Acción: GIRAR A LA IZQUIERDA")
            elif obj['position'] == 'CENTRO':
                print("    → Acción: AVANZAR")
            else:
                print("    → Acción: GIRAR A LA DERECHA")
    else:
        print("Sin objetos detectados - ESPERANDO...")
    
    # Cerrar al terminar
    # detector.close()


# ============================================================================
# ESTRUCTURA DE OUTPUT
# ============================================================================
#
# result = detector.get_detection()
#
# result contiene:
#   {
#       'detected': bool,              # ¿Hay objetos?
#       'objects': [                   # Lista de objetos detectados
#           {
#               'class': 0,                    # Clase (0 = persona)
#               'position': 'IZQUIERDA',       # IZQUIERDA, CENTRO, o DERECHA
#               'confidence': 0.92,            # Confianza 0.0-1.0
#               'track_id': 1,                 # ID de tracking
#               'bbox': (x1, y1, x2, y2),      # Bounding box
#               'center': (cx, cy)             # Centro del objeto
#           },
#           ...
#       ],
#       'frame': np.ndarray,            # Frame capturado
#       'fps': 15.2,                    # FPS actual
#       'timestamp': 1699123456.789     # Timestamp Unix
#   }


# ============================================================================
# REFERENCIA DE PARÁMETROS
# ============================================================================
#
# --camera [int]
#   Índice de cámara (0 = primera cámara)
#   Default: 0
#   Ej: --camera 1
#
# --model [str]
#   Modelo YOLOv8 a usar
#   Opciones: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), etc.
#   Default: yolov8n.pt
#   Ej: --model yolov8s.pt
#
# --conf [float]
#   Confianza mínima para considerar una detección válida
#   Rango: 0.0 - 1.0
#   Default: 0.5
#   Ej: --conf 0.6
#
# --imgsz [int]
#   Tamaño de imagen para inferencia (menor = más rápido pero menos preciso)
#   Valores típicos: 320, 416, 640
#   Default: 320
#   Ej: --imgsz 416
#
# --width [int]
#   Ancho de captura de la cámara
#   Default: 640
#   Ej: --width 320
#
# --height [int]
#   Alto de captura de la cámara
#   Default: 480
#   Ej: --height 240
#
# --fps [int]
#   Frames por segundo para captura de cámara
#   Default: 30
#   Ej: --fps 15
#
# --tracker [str]
#   Tipo de tracker a usar
#   Default: bytetrack.yaml
#   Ej: --tracker botsort.yaml
#
# --half
#   Usar half precision (FP16) para reducir consumo de memoria
#   Solo funciona con GPU (CUDA)
#   Ej: --half
#
# --arm-optimize
#   Activar optimizaciones para ARM/Raspberry Pi
#   Ej: --arm-optimize
#
# --verbose
#   Mostrar mensajes detallados de debug
#   Ej: --verbose


# ============================================================================
# OPTIMIZACIONES DISPONIBLES
# ============================================================================
#
# Con --arm-optimize se activan automáticamente:
#
# 1. OpenCV Optimization:
#    - setNumThreads(2) para mejor uso de recursos
#    - setUseOptimized(True) para usar instrucciones NEON si están disponibles
#
# 2. Detección automática de Raspberry Pi:
#    - Detecta si está ejecutándose en Raspberry Pi
#    - Muestra temperatura del CPU
#    - Configura automáticamente optimizaciones
#
# 3. Soporte NCNN:
#    - Inferencia ultra-rápida (compilación manual requerida)
#    - Use: --use-ncnn
#
# 4. Monitoreo de recursos:
#    - Muestra FPS en tiempo real
#    - Monitoreo de temperatura en Raspberry Pi
#    - Información de carga


# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
# Si el detector es lento en Raspberry Pi:
#   1. Reducir resolución: --width 320 --height 240
#   2. Reducir tamaño de imagen: --imgsz 320
#   3. Usar modelo nano: --model yolov8n.pt
#   4. Reducir FPS: --fps 15
#   5. Activar FP16: --half (si tienes GPU)
#
# Si faltan detecciones:
#   1. Aumentar confianza: --conf 0.3 (menos restrictivo)
#   2. Aumentar tamaño de imagen: --imgsz 416
#   3. Usar modelo más grande: --model yolov8s.pt
#
# Si hay errores de importación:
#   1. Verificar que los paquetes estén instalados:
#      pip install ultralytics opencv-python torch numpy
#   2. En Raspberry Pi, para mejor soporte:
#      pip install ncnn
#
# Si se congela la cámara:
#   1. Reducir resolución
#   2. Usar thread separado para captura
#   3. Verificar conexión USB de la cámara


# ============================================================================
# COMANDOS RÁPIDOS
# ============================================================================

# Modo básico (Raspberry Pi):
# python yolo-detection-arm.py --camera 0 --model yolov8n.pt --arm-optimize --verbose

# Modo performance (máxima velocidad):
# python yolo-detection-arm.py --camera 0 --model yolov8n.pt --imgsz 320 --width 320 --height 240 --arm-optimize

# Modo calidad (mejor precisión):
# python yolo-detection-arm.py --camera 0 --model yolov8m.pt --imgsz 416 --width 640 --height 480

# Con GPU (FP16):
# python yolo-detection-arm.py --camera 0 --model yolov8n.pt --half

# Debug completo:
# python yolo-detection-arm.py --camera 0 --model yolov8n.pt --arm-optimize --verbose --conf 0.3

print(__doc__)
