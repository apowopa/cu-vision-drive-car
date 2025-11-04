# Car Control - Control de Carrito Autónomo

Scripts para controlar el carrito siguiendo objetos detectados por el tracker YOLOv8 con soporte para sensores de distancia.

## 📂 Estructura de Archivos

```
car-control/
├── controlador_example.py              # Control básico de motores + sensores VL53L0X
├── tracker_controller.py               # Controlador que integra tracker + motores
├── tracker_controller_advanced.py      # Versión avanzada con sensores de obstáculos
├── ejemplo_tracker_controller.py       # Ejemplos de uso interactivos
├── README.md                           # Este archivo
└── README_TRACKER.md                   # Documentación del tracker_controller
```

## 🎯 Scripts Disponibles

### 1. `controlador_example.py` - Control Básico

Script de ejemplo que muestra cómo controlar los motores y leer sensores de distancia.

**Características:**
- Control de 2 motores (izquierdo y derecho)
- Lectura de 3 sensores VL53L0X (frente, izquierda, derecha)
- Lógica básica de evitar obstáculos

**Uso:**
```bash
python controlador_example.py
```

### 2. `tracker_controller.py` - Seguimiento Automático ⭐

**Este es el script principal** que integra el tracker YOLOv8 con el control de motores.

**Características:**
- Detecta objetos con YOLOv8
- Controla automáticamente el carrito para seguir objetos
- Soporte para NCNN (más rápido en ARM)
- Modo simulación para pruebas sin hardware

**Uso:**
```bash
# Simulación (sin hardware)
python tracker_controller.py --simulation

# Hardware real
python tracker_controller.py --arm-optimize

# Con NCNN (Raspberry Pi)
python tracker_controller.py --use-ncnn --arm-optimize

# Seguir autos en lugar de personas
python tracker_controller.py --class 2 --arm-optimize
```

### 3. `tracker_controller_advanced.py` - Tracker + Sensores

Versión avanzada que combina tracking + sensores de distancia.

**Características:**
- Sigue objetos detectados
- Evita obstáculos usando sensores VL53L0X
- Control automático de velocidad y dirección

**Uso:**
```bash
python tracker_controller_advanced.py --arm-optimize

# Con distancia crítica personalizada
python tracker_controller_advanced.py --obstacle-distance 300 --arm-optimize
```

### 4. `ejemplo_tracker_controller.py` - Ejemplos Interactivos

Script con menú interactivo que demuestra varios casos de uso.

**Uso:**
```bash
# Menú interactivo
python ejemplo_tracker_controller.py

# O ejecutar ejemplo directo
python ejemplo_tracker_controller.py 1    # Simulación
python ejemplo_tracker_controller.py 2    # Personas
python ejemplo_tracker_controller.py 3    # Autos con NCNN
```

## 🚀 Inicio Rápido

### 1️⃣ Probar sin hardware (modo simulación)

```bash
cd /home/apowo/Projects/cu-vision-drive-car/car-control
python tracker_controller.py --simulation --verbose
```

**Resultado esperado:**
```
[INFO] Inicializando TrackerController...
[INFO] Inicializando detector YOLOv8...
[INFO] Inicializando controlador de motores...
[WARNING] gpiozero no disponible. Modo simulación.
[INFO] TrackerController listo!

✓ CENTRO     | Confianza: 0.92
⬆️  FORWARD
```

### 2️⃣ Con hardware real (Raspberry Pi)

```bash
python tracker_controller.py --arm-optimize
```

### 3️⃣ Con máximas optimizaciones (Raspberry Pi 4)

```bash
python tracker_controller.py \
    --use-ncnn \
    --arm-optimize \
    --class 0 \
    --verbose
```

## 📊 Comparación de Scripts

| Feature | basic | tracker | advanced |
|---------|-------|---------|----------|
| Motores | ✓ | ✓ | ✓ |
| Sensores VL53L0X | ✓ | ✗ | ✓ |
| Tracker YOLOv8 | ✗ | ✓ | ✓ |
| Seguimiento automático | ✗ | ✓ | ✓ |
| Evitar obstáculos | ✓ (manual) | ✗ | ✓ (automático) |
| NCNN support | ✗ | ✓ | ✓ |
| ARM optimize | ✗ | ✓ | ✓ |

## 🎮 Flujo de Trabajo Recomendado

### Para Principiantes:
1. Iniciar con `ejemplo_tracker_controller.py` opción 1 (simulación)
2. Probar con hardware: `tracker_controller.py --arm-optimize`
3. Optimizar: `tracker_controller.py --use-ncnn --arm-optimize`

### Para Avanzados:
1. Usar `tracker_controller_advanced.py` para máximo control
2. Personalizar `process_detection()` con tu lógica
3. Integrar con sensores adicionales

## ⚙️ Parámetros Comunes

```bash
# Tracker
--model yolov8n.pt          # Modelo YOLOv8
--camera 0                  # Índice de cámara
--class 0                   # Clase COCO (0=personas, 2=autos, 16=perros)
--conf 0.5                  # Confianza mínima

# Optimización
--use-ncnn                  # Usar NCNN (ARM/Raspberry Pi)
--arm-optimize              # Optimizaciones ARM

# Modo
--simulation                # Modo simulación (sin hardware)
--verbose                   # Debug info
--delay 0.1                 # Delay entre iteraciones
--max-iterations 100        # Máx iteraciones
```

## 🔧 Control de Motores

Todos los controladores incluyen estas acciones:

```python
motor.forward()    # Avanza recto
motor.backward()   # Retrocede
motor.turn_left()  # Gira izquierda
motor.turn_right() # Gira derecha
motor.stop()       # Detiene
motor.brake()      # Frena (motores activos)
```

## 🔌 Hardware Requerido

### Motores
- 2 motores DC con control GPIO
- GPIO 5, 6 (motor izquierdo)
- GPIO 13, 19 (motor derecho)

### Sensores (opcional para tracker_controller_advanced)
- 3 sensores VL53L0X (ToF distance)
- GPIO 4, 17, 27 (XSHUT pins)
- Direcciones I2C: 0x30, 0x31, 0x32

### Cámara
- Cámara USB o Raspberry Pi Camera

## 📚 Ejemplos Prácticos

### Seguir personas

```bash
python tracker_controller.py \
    --class 0 \
    --arm-optimize \
    --verbose
```

### Seguir autos con NCNN (Raspberry Pi)

```bash
python tracker_controller.py \
    --class 2 \
    --use-ncnn \
    --arm-optimize
```

### Prueba rápida de 50 iteraciones

```bash
python tracker_controller.py \
    --simulation \
    --max-iterations 50
```

### Modo avanzado con sensores

```bash
python tracker_controller_advanced.py \
    --obstacle-distance 300 \
    --arm-optimize \
    --verbose
```

## 🐛 Troubleshooting

### "gpiozero no disponible"
El script automáticamente cambia a modo simulación si no puedes instalar gpiozero.

```bash
# Para instalar gpiozero en Raspberry Pi:
sudo apt-get install -y python3-gpiozero python3-rpi-lgpio
```

### FPS bajo
```bash
# Usar NCNN
python tracker_controller.py --use-ncnn

# Reducir delay
python tracker_controller.py --delay 0.05

# Usar modelo más pequeño
python tracker_controller.py --model yolov8n.pt
```

### No se detectan objetos
```bash
# Reducir confianza mínima
python tracker_controller.py --conf 0.3

# Ver debug info
python tracker_controller.py --verbose
```

## 📖 Integración con Navegación

Ver `navigation_integration_example.py` en la carpeta raíz para integración completa con el sistema de navegación del carrito.

## 🔗 Archivos Relacionados

- `camera-detection/yolo-detection-arm.py` - Detector base
- `camera-detection/convert_yolo_to_ncnn.py` - Conversión a NCNN
- `EJEMPLOS_USO.md` - Ejemplos del detector
- `navigation_integration_example.py` - Integración completa

## 📝 Notas

- Todos los scripts soportan Ctrl+C para salir gracefully
- Los logs se muestran en tiempo real
- Modo simulación es perfecto para testing sin hardware
- NCNN ofrece 2-3x velocidad en Raspberry Pi

## 🎓 Aprender Más

Ver archivos README específicos:
- `README_TRACKER.md` - Documentación detallada del tracker_controller

O revisar el código:
- `tracker_controller.py` - Clase principal bien documentada
- `ejemplo_tracker_controller.py` - 5 ejemplos completos
