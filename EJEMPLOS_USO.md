#!/usr/bin/env python3
"""
Ejemplos de uso del detector YOLOv8 con soporte NCNN para ARM
"""

# Ejemplos de uso del detector YOLOv8 con soporte NCNN para ARM

## EJEMPLO 1: Modo simple YOLOv8 (sin NCNN)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --conf 0.5 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 2: Modo NCNN automático (convierte y usa NCNN si está disponible)

# Ejemplos de uso del detector YOLOv8 con soporte NCNN para ARM

## EJEMPLO 1: Modo simple YOLOv8 (sin NCNN)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --conf 0.5 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 2: Modo NCNN automático (convierte y usa NCNN si está disponible)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --use-ncnn \
    --arm-optimize \
    --verbose
```

## EJEMPLO 3: Buscar solo personas (clase 0)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 0 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 4: Buscar solo bicicletas (clase 1)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 1 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 5: Buscar autos (clase 2)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 2 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 6: Optimizado para Raspberry Pi 4 (performance máximo)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 0 \
    --imgsz 320 \
    --width 320 \
    --height 240 \
    --fps 15 \
    --use-ncnn \
    --arm-optimize \
    --verbose
```

## EJEMPLO 7: Con GPU (si tienes Jetson Nano o similar)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --half \
    --verbose
```

## EJEMPLO 8: Usar como módulo en tu código

```python
from camera_detection.yolo_detection_arm import ObjectDetector

# Crear detector buscando solo personas (clase 0)
detector = ObjectDetector(
    model_path="yolov8n.pt",
    camera_idx=0,
    conf_threshold=0.5,
    target_class=0,  # <-- Especificar clase objetivo
    imgsz=320,
    width=640,
    height=480,
    fps=30,
    tracker="bytetrack.yaml",
    arm_optimize=True,
    use_ncnn=True,  # Habilita conversión automática a NCNN
    verbose=True
)

# Loop de detección
while True:
    result = detector.get_detection()
    
    if result['detected']:
        print(f"FPS: {result['fps']:.1f}")
        print(f"Mode: {'NCNN' if detector.use_ncnn_mode else 'YOLOv8'}")
        
        for obj in result['objects']:
            print(f"  - {obj['position']}: confianza {obj['confidence']:.2f}")
            print(f"    Track ID: {obj['track_id']}")
            print(f"    Centro: {obj['center']}")
    else:
        print("Sin objetos detectados")

detector.close()
```

## CLASES COCO DISPONIBLES

| ID | Nombre | ID | Nombre |
|----|--------|----|----|
| 0 | person | 20 | traffic light |
| 1 | bicycle | 21 | fire hydrant |
| 2 | car | 22 | stop sign |
| 3 | motorcycle | 23 | parking meter |
| 4 | airplane | 24 | bench |
| 5 | bus | 25 | cat |
| 6 | train | 26 | dog |
| 7 | truck | 27 | horse |
| 8 | boat | 28 | sheep |
| 9 | traffic light | 29 | cow |
| 10 | fire hydrant | 30 | elephant |
| 11 | stop sign | 31 | bear |
| 12 | parking meter | 32 | zebra |
| 13 | bench | 33 | giraffe |
| 14 | cat | 34 | backpack |
| 15 | dog | 35 | umbrella |
| 16 | horse | 36 | handbag |
| 17 | sheep | 37 | tie |
| 18 | cow | 38 | suitcase |
| 19 | elephant | 39 | frisbee |

## CARACTERÍSTICAS PRINCIPALES

### 1. Parámetro `--class` para filtrar detecciones

Usa el parámetro `--class` seguido del ID de la clase COCO:

```bash
# Solo personas (clase 0)
python yolo-detection-arm.py --camera 0 --class 0

# Solo autos (clase 2)
python yolo-detection-arm.py --camera 0 --class 2

# Solo perros (clase 16)
python yolo-detection-arm.py --camera 0 --class 16
```

### 2. Soporte automático NCNN

- Si `--use-ncnn` está activado, convierte automáticamente el modelo YOLOv8 a NCNN
- Si NCNN no está disponible, cae atrás automáticamente a YOLOv8
- El modo actual se muestra en pantalla ("NCNN" o "YOLOv8")

### 3. Detección de 3 zonas

La pantalla se divide en 3 zonas horizontales:

- **IZQUIERDA** (rojo): x < ancho/3
- **CENTRO** (verde): ancho/3 <= x < 2*ancho/3
- **DERECHA** (azul): x >= 2*ancho/3

### 4. Tracking de objetos

- Track IDs persistentes entre frames
- Compatible con ByteTrack (seguimiento robusto)

### 5. Optimizaciones ARM

- Detección automática de Raspberry Pi
- Configuración de threads optimizada para ARM
- NEON Optimization activada automáticamente
- Monitoreo de temperatura del CPU

### 6. Estructura del output

```python
result = {
    'detected': bool,              # ¿Hay objetos?
    'objects': [                   # Lista de objetos detectados
        {
            'class': 0,                    # Clase (0 = persona, 2 = auto, etc)
            'position': 'IZQUIERDA',       # IZQUIERDA/CENTRO/DERECHA
            'confidence': 0.92,            # Confianza 0.0-1.0
            'track_id': 1,                 # ID de tracking
            'bbox': (x1, y1, x2, y2),      # Bounding box
            'center': (cx, cy)             # Centro del objeto
        }
    ],
    'frame': frame,                # Frame capturado (numpy array)
    'fps': 15.2,                   # FPS actual
    'timestamp': 1699123456.789    # Timestamp Unix
}
```

## CONVERSIÓN MANUAL A NCNN (sin usar el script)

Si prefieres convertir el modelo manualmente por separado:

```bash
python camera-detection/convert_yolo_to_ncnn.py \
    --input yolov8n.pt \
    --output yolov8n_ncnn \
    --imgsz 320
```

Esto genera dos archivos:
- `yolov8n_ncnn.param` - Parámetros del modelo
- `yolov8n_ncnn.bin` - Pesos del modelo

## TROUBLESHOOTING

### Q: "NCNN no está disponible"

A: Instala NCNN manualmente. La compilación desde fuente es recomendada en ARM:

```bash
pip install ncnn
# O compilar desde fuente para mejor performance en Raspberry Pi
```

### Q: ¿Cuándo debo usar NCNN?

A: 
- **Raspberry Pi/ARM**: Usa NCNN para 2-3x más velocidad que PyTorch
- **Jetson Nano/Xavier**: No hay beneficio significativo vs YOLOv8
- **GPU potente**: Usa YOLOv8 normal, NCNN es para CPU

### Q: "Conversión a NCNN muy lenta"

A: Es normal - solo ocurre una vez al iniciar. El modelo convertido se reutiliza en siguientes ejecuciones.

### Q: "FPS bajo en Raspberry Pi 4"

A: Intenta estas optimizaciones:

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --class 0 \
    --imgsz 320 \
    --width 320 \
    --height 240 \
    --fps 15 \
    --use-ncnn \
    --arm-optimize \
    --verbose
```

**Pasos de debugging:**
1. Reducir resolución: `--width 320 --height 240`
2. Reducir tamaño de imagen: `--imgsz 320`
3. Habilitar NCNN: `--use-ncnn`
4. Reducir FPS objetivo: `--fps 15`

### Q: ¿Cómo integro esto con navegación?

A: Ver ejemplo 8 arriba. Usa `get_detection()` en tu loop principal:

```python
result = detector.get_detection()

if result['detected']:
    for obj in result['objects']:
        if obj['position'] == 'IZQUIERDA':
            # Girar a la izquierda
            pass
        elif obj['position'] == 'DERECHA':
            # Girar a la derecha
            pass
        else:
            # Centro - seguir recto
            pass
```

### Q: "¿Cuáles son los valores por defecto?"

A:
- `--camera`: 0
- `--class`: 0 (personas)
- `--conf`: 0.5
- `--imgsz`: 640
- `--width`: 640
- `--height`: 480
- `--fps`: 30
- `--tracker`: bytetrack.yaml
- `--use-ncnn`: desactivado
- `--arm-optimize`: desactivado

### Q: "¿Cómo cambio el modelo (pequeño/mediano/grande)?"

A:
```bash
# Nano (más rápido, menos preciso)
python yolo-detection-arm.py --model yolov8n.pt

# Small (balance)
python yolo-detection-arm.py --model yolov8s.pt

# Medium (más preciso, más lento)
python yolo-detection-arm.py --model yolov8m.pt

# Large (muy preciso, muy lento)
python yolo-detection-arm.py --model yolov8l.pt
```

## CASOS DE USO

### Detectar solo personas

```bash
python yolo-detection-arm.py --camera 0 --class 0 --arm-optimize
```

### Detectar solo vehículos (autos, buses, camiones)

```bash
# Solo autos
python yolo-detection-arm.py --camera 0 --class 2

# Solo buses
python yolo-detection-arm.py --camera 0 --class 5

# Solo camiones
python yolo-detection-arm.py --camera 0 --class 7
```

### Detectar animales

```bash
# Perros
python yolo-detection-arm.py --camera 0 --class 16

# Gatos
python yolo-detection-arm.py --camera 0 --class 15

# Pájaros (no disponible en COCO, usar clase 0 como fallback)
```

### Modo balanceado para Raspberry Pi

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 0 \
    --imgsz 416 \
    --width 416 \
    --height 416 \
    --fps 20 \
    --use-ncnn \
    --arm-optimize
```

## VARIABLES DE ENTORNO

Puedes configurar comportamientos adicionales:

```bash
# Mostrar información de debug (requiere --verbose)
export DEBUG=1

# Forzar CPU (aunque haya GPU disponible)
export CUDA_VISIBLE_DEVICES=""

# Configurar threads de OpenCV
export OPENCV_THREADS_NUM=2
```

## INTEGRACIÓN CON NAVIGATION

El detector está diseñado para integrase con el sistema de navegación:

```python
from camera_detection.yolo_detection_arm import ObjectDetector
from car_control.controlador import CarController

detector = ObjectDetector(target_class=0)  # Detectar personas
controller = CarController()

while True:
    result = detector.get_detection()
    
    if result['detected']:
        obj = result['objects'][0]  # Primera persona detectada
        
        if obj['position'] == 'IZQUIERDA':
            controller.turn_left()
        elif obj['position'] == 'DERECHA':
            controller.turn_right()
        else:
            controller.move_forward()
```

Ver `navigation_integration_example.py` para más detalles.

## EJEMPLO 3: Buscar solo personas (clase 0)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 0 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 4: Buscar solo bicicletas (clase 1)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 1 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 5: Buscar autos (clase 2)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 2 \
    --arm-optimize \
    --verbose
```

## EJEMPLO 6: Optimizado para Raspberry Pi 4 (performance máximo)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 0 \
    --imgsz 320 \
    --width 320 \
    --height 240 \
    --fps 15 \
    --use-ncnn \
    --arm-optimize \
    --verbose
```

## EJEMPLO 7: Con GPU (si tienes Jetson Nano o similar)

```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --half \
    --verbose
```

## EJEMPLO 8: Usar como módulo en tu código


# ============================================================================
# EJEMPLO 2: Buscar vehículos (clase 2 - car)
# ============================================================================

python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 2 \
    --conf 0.5 \
    --arm-optimize \
    --verbose


# ============================================================================
# EJEMPLO 3: Modo NCNN automático - Buscar bicicletas (clase 1)
# ============================================================================

python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --class 1 \
    --use-ncnn \
    --arm-optimize \
    --verbose

# Loop de detección
while True:
    result = detector.get_detection()
    
    if result['detected']:
        print(f"FPS: {result['fps']:.1f}")
        print(f"Mode: {'NCNN' if detector.use_ncnn_mode else 'YOLOv8'}")
        
        for obj in result['objects']:
            print(f"  - {obj['position']}: confianza {obj['confidence']:.2f}")
            print(f"    Track ID: {obj['track_id']}")
            print(f"    Centro: {obj['center']}")
    else:
        print("Sin objetos detectados")
    
    # break  # Descomentar para salir después de una iteración

# detector.close()


# ============================================================================
# CARACTERÍSTICAS PRINCIPALES
# ============================================================================

# 1. Soporte automático NCNN:
#    - Si --use-ncnn está activado, convierte automáticamente el modelo
#    - Si NCNN no está disponible, cae atrás a YOLOv8
#    - Modo se muestra en pantalla ("NCNN" o "YOLOv8")

# 2. Detección de 3 zonas:
#    - IZQUIERDA (rojo): x < ancho/3
#    - CENTRO (verde): ancho/3 <= x < 2*ancho/3
#    - DERECHA (azul): x >= 2*ancho/3

# 3. Tracking de objetos:
#    - Track IDs persistentes
#    - Compatible con ByteTrack

# 4. Optimizaciones ARM:
#    - Detección automática de Raspberry Pi
#    - Configuración de threads optimizada
#    - NEON Optimization
#    - Monitoreo de temperatura del CPU

# 5. Output estructura:
result = {
    'detected': bool,              # ¿Hay objetos?
    'objects': [                   # Lista de objetos
        {
            'class': 0,                    # Clase (0 = persona)
            'position': 'IZQUIERDA',       # IZQUIERDA/CENTRO/DERECHA
            'confidence': 0.92,            # Confianza 0.0-1.0
            'track_id': 1,                 # ID de tracking
            'bbox': (x1, y1, x2, y2),      # Bounding box
            'center': (cx, cy)             # Centro del objeto
        }
    ],
    'frame': frame,                # Frame capturado
    'fps': 15.2,                   # FPS actual
    'timestamp': 1699123456.789    # Timestamp Unix
}


# ============================================================================
# CONVERSIÓN MANUAL A NCNN (sin usar el script)
# ============================================================================

# Si prefieres convertir el modelo por separado:

python convert_yolo_to_ncnn.py \
    --input yolov8n.pt \
    --output yolov8n_ncnn \
    --imgsz 320


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Q: "NCNN no está disponible"
# A: Instala NCNN manualmente (compilación recomendada para ARM):
#    pip install ncnn

# Q: ¿Cuándo debo usar NCNN?
# A: En Raspberry Pi para 2-3x más velocidad que PyTorch
#    En GPU (Jetson), no hay beneficio - usa YOLOv8 normal

# Q: "Conversión a NCNN muy lenta"
# A: Es normal, solo ocurre una sola vez. Luego se reutiliza el modelo convertido

# Q: "FPS bajo en Raspberry Pi 4"
# A: Prueba:
#    1. Reducir resolución: --width 320 --height 240
#    2. Reducir imgsz: --imgsz 320
#    3. Usar NCNN: --use-ncnn
#    4. Reducir FPS: --fps 15

# Q: ¿Cómo integro esto con navegación?
# A: Ver ejemplo 5 arriba, usa get_detection() en tu loop principal
#    Toma decisiones basadas en obj['position']
