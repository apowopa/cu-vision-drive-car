# Tracker Controller - Seguimiento Automático con Control de Carrito

Script que integra la detección YOLOv8 con el control de motores del carrito. El carrito gira automáticamente para seguir objetos detectados.

## 🎯 Funcionamiento

El controlador recibe detecciones del tracker YOLOv8 y controla los motores según la posición del objeto:

```
IZQUIERDA    →    Gira a la izquierda
CENTRO       →    Avanza recto
DERECHA      →    Gira a la derecha
Sin objeto   →    Se detiene
```

## 📋 Requisitos

```bash
# Dependencias de Python
pip install gpiozero
pip install lgpio

# O usar la raspberry tool para instalar:
# sudo apt install -y python3-gpiozero python3-rpi-lgpio

# El detector YOLOv8 debe estar disponible
# Ver: camera-detection/yolo-detection-arm.py
```

## 🚀 Uso Básico

### Modo simulación (sin hardware)

Perfecto para probar sin tener el carrito conectado:

```bash
python tracker_controller.py --simulation --verbose
```

Salida ejemplo:
```
[INFO] Inicializando TrackerController...
[INFO] Inicializando detector YOLOv8...
[INFO] Inicializando controlador de motores...
[INFO] TrackerController listo!
[INFO] Iniciando loop de seguimiento...

✓ CENTRO     | Confianza: 0.92
   FPS promedio: 18.5
✓ IZQUIERDA  | Confianza: 0.87
   FPS promedio: 18.3
✗ Sin detecciones
   FPS promedio: 19.1
```

### Usar con hardware real

```bash
python tracker_controller.py --arm-optimize
```

### Optimizado para Raspberry Pi

```bash
python tracker_controller.py \
    --arm-optimize \
    --use-ncnn \
    --verbose
```

## 📚 Ejemplos Completos

### EJEMPLO 1: Seguimiento básico de personas

```bash
python tracker_controller.py \
    --class 0 \
    --simulation
```

### EJEMPLO 2: Seguimiento de autos

```bash
python tracker_controller.py \
    --class 2 \
    --arm-optimize
```

### EJEMPLO 3: Seguimiento de perros

```bash
python tracker_controller.py \
    --class 16 \
    --arm-optimize \
    --use-ncnn
```

### EJEMPLO 4: Con velocidad controlada

Para paradas más lentas, usar delay:

```bash
python tracker_controller.py \
    --delay 0.2 \
    --arm-optimize \
    --verbose
```

### EJEMPLO 5: Pruebas limitadas (100 iteraciones)

```bash
python tracker_controller.py \
    --max-iterations 100 \
    --simulation \
    --verbose
```

### EJEMPLO 6: Hardware real con todas las optimizaciones

```bash
python tracker_controller.py \
    --model yolov8n.pt \
    --camera 0 \
    --class 0 \
    --conf 0.5 \
    --arm-optimize \
    --use-ncnn \
    --verbose
```

## ⚙️ Parámetros

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--model` | Ruta del modelo YOLOv8 | yolov8n.pt |
| `--camera` | Índice de cámara | 0 |
| `--class` | Clase COCO a detectar | 0 (personas) |
| `--conf` | Umbral de confianza | 0.5 |
| `--use-ncnn` | Usar NCNN (más rápido en ARM) | false |
| `--arm-optimize` | Optimizaciones ARM/Raspberry Pi | false |
| `--simulation` | Modo simulación (sin hardware) | false |
| `--verbose` | Mostrar debug info | false |
| `--max-iterations` | Máx iteraciones (infinito si omite) | None |
| `--delay` | Delay entre iteraciones (seg) | 0.1 |

## 🔧 Control de Motores

El controlador de motores soporta las siguientes acciones:

```python
motor_controller.forward()   # Avanza
motor_controller.backward()  # Retrocede
motor_controller.turn_left() # Gira izquierda
motor_controller.turn_right()# Gira derecha
motor_controller.stop()      # Detiene
motor_controller.brake()     # Frena (motores activos)
```

## 📖 Uso como Módulo

Puedes importar y usar en tu propio código:

```python
from tracker_controller import TrackerController, MotorController

# Crear controlador
controller = TrackerController(
    model_path="yolov8n.pt",
    camera_idx=0,
    target_class=0,  # Detectar personas
    arm_optimize=True,
    use_ncnn=True,
    simulation=False,  # Hardware real
    verbose=True
)

# Ejecutar
try:
    controller.run(max_iterations=1000, delay=0.1)
except KeyboardInterrupt:
    print("Detenido por usuario")
finally:
    controller.cleanup()
```

## 🎮 Control Manual

Si prefieres usar solo el controlador de motores:

```python
from tracker_controller import MotorController

motor = MotorController(use_simulation=False)

motor.forward()
time.sleep(2)
motor.turn_left()
time.sleep(1)
motor.stop()
```

## 🐛 Troubleshooting

### "gpiozero no disponible"

```bash
# Instalar dependencias
sudo apt-get install -y python3-gpiozero python3-rpi-lgpio

# O usar pip
pip install gpiozero
```

### Cambiando a modo simulación automáticamente

Si no tienes el hardware conectado, el script automáticamente cambia a simulación:

```
[WARNING] gpiozero no disponible. Modo simulación.
[ERROR] No se pueden inicializar los motores: ...
[INFO] Cambiando a modo simulación.
```

### FPS bajo

```bash
python tracker_controller.py \
    --delay 0.2 \
    --arm-optimize \
    --use-ncnn
```

### Detecciones inconsistentes

Aumentar confianza mínima:

```bash
python tracker_controller.py \
    --conf 0.7 \
    --verbose
```

## 🔌 Hardware

### Pines GPIO utilizados

```
Motor Izquierdo:
  - GPIO 5  (motor_left_a)
  - GPIO 6  (motor_left_b)

Motor Derecho:
  - GPIO 13 (motor_right_a)
  - GPIO 19 (motor_right_b)
```

### Cinemática del carrito

| Estado | Motor Izq (a,b) | Motor Der (a,b) | Resultado |
|--------|-----------------|-----------------|-----------|
| FORWARD | (1,0) | (1,0) | Avanza |
| BACKWARD | (0,1) | (0,1) | Retrocede |
| LEFT | (0,1) | (1,0) | Gira izq |
| RIGHT | (1,0) | (0,1) | Gira der |
| STOP | (0,0) | (0,0) | Detiene |
| BRAKE | (1,1) | (1,1) | Frena |

## 📝 Notas

- El script usa **ByteTrack** para seguimiento robusto entre frames
- Las detecciones se filtran por clase COCO (0-79)
- Soporta tanto YOLOv8 nativo como NCNN
- En Raspberry Pi, NCNN ofrece 2-3x más velocidad
- El modo simulación es perfecto para pruebas sin hardware

## 🔄 Integración con Sensores

Si quieres agregar sensores (VL53L0X), puedes modificar `process_detection()`:

```python
def process_detection(self, result):
    """Procesa detección y sensor"""
    
    # Leer sensor de distancia
    distances = self.sensor_controller.read_sensors()
    dist_front = distances['frontal']
    
    # Si hay obstáculo muy cerca, detener
    if dist_front < 100:  # 100mm
        self.motor_controller.brake()
        return
    
    # Sino, seguir al objeto
    if result['detected']:
        obj = result['objects'][0]
        position = obj['position']
        
        if position == "IZQUIERDA":
            self.motor_controller.turn_left()
        elif position == "DERECHA":
            self.motor_controller.turn_right()
        else:
            self.motor_controller.forward()
    else:
        self.motor_controller.stop()
```

## 📦 Estructura de Archivos

```
car-control/
├── controlador_example.py      # Controlador con sensores (referencia)
├── tracker_controller.py       # Este archivo
└── README_TRACKER.md           # Esta documentación

camera-detection/
└── yolo-detection-arm.py       # Detector YOLOv8 (requerido)
```

## 🎓 Para Aprender Más

- Ver `EJEMPLOS_USO.md` para ejemplos del tracker
- Ver `navigation_integration_example.py` para integración avanzada
- Ver `controlador_example.py` para uso de sensores de distancia
