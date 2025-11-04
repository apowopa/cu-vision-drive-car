# Detector de Objetos YOLOv8 con Optimizaciones ARM/Raspberry Pi

## 🚀 Inicio Rápido

### Modo Standalone (Visualización en Tiempo Real)

```bash
python yolo-detection-arm.py --camera 0 --model yolov8n.pt --arm-optimize --verbose
```

**Controles:**
- `q` o `ESC`: Salir
- `ESPACIO`: Pausar/Reanudar

### Integración con Script de Navegación

```bash
python navigation_integration_example.py integration
```

### Usar como Módulo en Tu Código

```python
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
            print(f"Objeto en: {obj['position']}")
            print(f"Confianza: {obj['confidence']:.2f}")
            print(f"Track ID: {obj['track_id']}")
```

## 📊 Output del Detector

Cada llamada a `detector.get_detection()` retorna:

```python
{
    'detected': bool,              # ¿Hay objetos detectados?
    'objects': [                   # Lista de objetos
        {
            'class': 0,                    # Clase (0 = persona)
            'position': 'IZQUIERDA',       # IZQUIERDA, CENTRO, DERECHA
            'confidence': 0.92,            # Confianza 0.0-1.0
            'track_id': 1,                 # ID de tracking
            'bbox': (x1, y1, x2, y2),      # Bounding box
            'center': (cx, cy)             # Centro del objeto
        }
    ],
    'frame': np.ndarray,            # Frame capturado
    'fps': 15.2,                    # FPS actual
    'timestamp': 1699123456.789     # Timestamp Unix
}
```

## ⚙️ Parámetros Disponibles

| Parámetro | Descripción | Default | Rango |
|-----------|-------------|---------|-------|
| `--camera` | Índice de cámara | 0 | 0, 1, 2... |
| `--model` | Modelo YOLOv8 | yolov8n.pt | yolov8n/s/m/l/x.pt |
| `--conf` | Confianza mínima | 0.5 | 0.0-1.0 |
| `--imgsz` | Tamaño de imagen | 320 | 320, 416, 640 |
| `--width` | Ancho captura | 640 | > 0 |
| `--height` | Alto captura | 480 | > 0 |
| `--fps` | FPS de cámara | 30 | > 0 |
| `--tracker` | Tipo de tracker | bytetrack.yaml | bytetrack/botsort |
| `--half` | Usar FP16 | false | flag |
| `--arm-optimize` | Optimizaciones ARM | false | flag |
| `--verbose` | Debug mode | false | flag |

## 🎯 Casos de Uso Recomendados

### Raspberry Pi 4 (Performance):
```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --imgsz 320 \
    --width 320 \
    --height 240 \
    --arm-optimize \
    --verbose
```

### Raspberry Pi 5 (Equilibrado):
```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8s.pt \
    --imgsz 416 \
    --width 640 \
    --height 480 \
    --arm-optimize
```

### Jetson Nano (Performance):
```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8n.pt \
    --imgsz 320 \
    --width 640 \
    --height 480 \
    --arm-optimize
```

### Desktop/Laptop (Calidad):
```bash
python yolo-detection-arm.py \
    --camera 0 \
    --model yolov8m.pt \
    --imgsz 640 \
    --width 1280 \
    --height 720 \
    --half
```

## 🔧 Optimizaciones ARM

Con `--arm-optimize` se activan automáticamente:

1. **OpenCV Optimization**
   - Configuración de threads optimizada (2 threads)
   - NEON Optimization si está disponible

2. **Detección Automática de Raspberry Pi**
   - Adapta configuración automáticamente
   - Monitorea temperatura del CPU

3. **Soporte NCNN (opcional)**
   - Inferencia ultra-rápida (compilación manual)
   - Uso: `--use-ncnn`

## 📈 Ejemplo: Integración con Sistema de Navegación

```python
from yolo_detection_arm import ObjectDetector

class RobotNavigator:
    def __init__(self):
        self.detector = ObjectDetector(
            model_path="yolov8n.pt",
            camera_idx=0,
            arm_optimize=True
        )
    
    def navigate(self):
        """Navega basándose en detecciones de objetos"""
        result = self.detector.get_detection()
        
        if not result['detected']:
            self.stop()
            return
        
        # Obtener objeto más confiable
        best_obj = max(result['objects'], key=lambda x: x['confidence'])
        
        if best_obj['position'] == 'IZQUIERDA':
            self.turn_left()
        elif best_obj['position'] == 'CENTRO':
            self.move_forward()
        else:
            self.turn_right()
    
    def turn_left(self):
        print("← Girando a la izquierda")
        # Tu código aquí
    
    def turn_right(self):
        print("→ Girando a la derecha")
        # Tu código aquí
    
    def move_forward(self):
        print("↑ Avanzando")
        # Tu código aquí
    
    def stop(self):
        print("⊙ Esperando")
        # Tu código aquí

# Uso
robot = RobotNavigator()
while True:
    robot.navigate()
```

## 🐛 Troubleshooting

### Detector muy lento en Raspberry Pi
1. Reducir resolución: `--width 320 --height 240`
2. Reducir tamaño de imagen: `--imgsz 320`
3. Usar modelo nano: `--model yolov8n.pt`
4. Reducir FPS: `--fps 15`

### Faltan detecciones
1. Bajar confianza: `--conf 0.3`
2. Aumentar tamaño de imagen: `--imgsz 416`
3. Usar modelo más grande: `--model yolov8s.pt`

### Errores de importación
```bash
pip install ultralytics opencv-python torch numpy
```

Para Raspberry Pi con mejor soporte:
```bash
pip install ncnn
```

### La cámara se congela
1. Reducir resolución
2. Reducir FPS: `--fps 15`
3. Verificar conexión USB

## 📚 Archivos Principales

- **`yolo-detection-arm.py`** - Script principal con clase `ObjectDetector`
- **`navigation_integration_example.py`** - Ejemplos de integración
- **`USAGE_GUIDE.py`** - Guía detallada de uso
- **`README.md`** - Este archivo

## 🎓 Ejemplos Adicionales

Ver `USAGE_GUIDE.py` para:
- Ejemplos de uso completo
- Estructura detallada de output
- Referencia completa de parámetros
- Troubleshooting avanzado

Ver `navigation_integration_example.py` para:
- Integración con sistemas de navegación
- Modo callback
- Ejemplos de control de robots
