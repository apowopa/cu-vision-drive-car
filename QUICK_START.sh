#!/usr/bin/env zsh
# Quick reference - Comandos listos para Raspberry Pi 5

# ============================================================================
# 🔥 MÁXIMA VELOCIDAD - Recomendado para Motor Tracking
# ============================================================================

# Detector solo (sin GUI, headless)
python camera-detection/yolo-detection-arm.py --rpi5-ultra-fast --verbose

# Motor Tracker con máxima optimización
python car-control/tracker_controller.py \
  --width 160 --height 120 --imgsz 128 \
  --arm-optimize --verbose

# Motor Tracker con frame skipping (aún más rápido)
python car-control/tracker_controller.py \
  --width 160 --height 120 --imgsz 128 \
  --skip-frames 1 --arm-optimize --verbose

# ============================================================================
# 📊 PRUEBAS DE FPS
# ============================================================================

# Test ultra-fast preset
python test_fps.py --preset ultra-fast --duration 30

# Test ultra-fast con frame skipping
python test_fps.py --preset ultra-fast --skip-frames 1 --duration 30

# Test preset balanced
python test_fps.py --preset balanced --duration 30

# ============================================================================
# 🎯 DETECCIÓN POR CLASE (COCO IDs)
# ============================================================================

# Personas (clase 0) - Predeterminado
python car-control/tracker_controller.py --class 0 --arm-optimize --verbose

# Perros (clase 16)
python car-control/tracker_controller.py --class 16 --arm-optimize --verbose

# Gatos (clase 15)
python car-control/tracker_controller.py --class 15 --arm-optimize --verbose

# Autos/Coches (clase 2)
python car-control/tracker_controller.py --class 2 --arm-optimize --verbose

# ============================================================================
# 🔧 VARIACIONES DE VELOCIDAD
# ============================================================================

# Ultra-Fast (160x120, imgsz=128)
python car-control/tracker_controller.py \
  --width 160 --height 120 --imgsz 128 \
  --arm-optimize --verbose

# Fast (240x180, imgsz=160)
python car-control/tracker_controller.py \
  --width 240 --height 180 --imgsz 160 \
  --arm-optimize --verbose

# Balanced (320x240, imgsz=192)
python car-control/tracker_controller.py \
  --width 320 --height 240 --imgsz 192 \
  --arm-optimize --verbose

# Extreme (128x96, imgsz=96) - Muy bajo pero más rápido
python car-control/tracker_controller.py \
  --width 128 --height 96 --imgsz 96 \
  --arm-optimize --verbose

# ============================================================================
# ⚙️ OPCIONES AVANZADAS
# ============================================================================

# Con confianza baja (más detecciones)
python car-control/tracker_controller.py \
  --conf 0.2 --arm-optimize --verbose

# Sin optimización ARM (para debug)
python car-control/tracker_controller.py \
  --verbose

# Modo simulación (sin hardware)
python car-control/tracker_controller.py \
  --simulation --verbose

# Solo N iteraciones
python car-control/tracker_controller.py \
  --max-iterations 100 --verbose

# ============================================================================
# 📱 SSH Headless (Recomendado para Raspberry Pi)
# ============================================================================

# Conectar por SSH
ssh pi@raspberrypi.local

# Una vez conectado, ejecutar:
python camera-detection/yolo-detection-arm.py --rpi5-ultra-fast --verbose

# O con tracker controller
python car-control/tracker_controller.py --arm-optimize --verbose

# ============================================================================
# 🐛 DEBUG Y TROUBLESHOOTING
# ============================================================================

# Ver temperatura CPU
vcgencmd measure_temp

# Monitor de recursos en tiempo real
top

# Ejecutar detector con máximo debug
python camera-detection/yolo-detection-arm.py \
  --rpi5-ultra-fast --verbose 2>&1 | tee debug.log

# Test de FPS largo (5 minutos)
python test_fps.py --preset ultra-fast --duration 300

# ============================================================================
# 🚀 INICIO RÁPIDO
# ============================================================================

# 1. Preparar
cd /home/apowo/Projects/cu-vision-drive-car

# 2. Test rápido (30 segundos)
python test_fps.py --preset ultra-fast --duration 30

# 3. Si FPS OK, ejecutar tracker
python car-control/tracker_controller.py --arm-optimize --verbose

# 4. Control del vehículo
# - IZQUIERDA: gira izquierda
# - CENTRO: avanza recto
# - DERECHA: gira derecha
# - Sin detección: se detiene

# ============================================================================
# 📈 PARÁMETROS CLAVE
# ============================================================================

# Resolución: 160x120 (bajo pero rápido)
# YOLO input: 128 (muy reducido)
# Max detectiones: 100
# FPS objetivo: 15
# Threads CV2: 1
# Threads PyTorch: 3 (cpu_count - 1)
# Esperado: 20-30 FPS

# ============================================================================
