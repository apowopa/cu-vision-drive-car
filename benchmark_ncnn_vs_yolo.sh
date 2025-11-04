#!/bin/zsh
# Test NCNN vs YOLOv8 en Raspberry Pi 5

cd /home/apowo/Projects/cu-vision-drive-car

echo "═══════════════════════════════════════════════════════"
echo "BENCHMARK: NCNN vs YOLOv8"
echo "═══════════════════════════════════════════════════════"
echo ""

# Test 1: YOLOv8 Nativo - Ultra-Fast
echo "🧪 TEST 1: YOLOv8 Nativo (160x120, imgsz=128)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python camera-detection/yolo-detection-arm.py \
  --width 160 --height 120 --imgsz 128 \
  --arm-optimize --verbose | head -20
echo ""
echo "Ejecutar: python test_fps.py --preset ultra-fast --duration 30"
echo ""

# Test 2: NCNN - Ultra-Fast
echo ""
echo "🧪 TEST 2: NCNN INT8 Quantized (160x120, imgsz=128)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python camera-detection/yolo-detection-arm.py \
  --width 160 --height 120 --imgsz 128 \
  --use-ncnn --arm-optimize --verbose | head -30
echo ""

# Test 3: Motor Tracker con YOLOv8
echo ""
echo "🧪 TEST 3: Motor Tracker - YOLOv8"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python car-control/tracker_controller.py \
  --width 160 --height 120 --imgsz 128 \
  --arm-optimize --simulation --verbose | head -20
echo ""

# Test 4: Motor Tracker con NCNN
echo ""
echo "🧪 TEST 4: Motor Tracker - NCNN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python car-control/tracker_controller.py \
  --width 160 --height 120 --imgsz 128 \
  --use-ncnn --arm-optimize --simulation --verbose | head -20
echo ""

echo "═══════════════════════════════════════════════════════"
echo "✅ TESTS COMPLETADOS"
echo ""
echo "Para pruebas de FPS continuas, usa:"
echo "  python test_fps.py --preset ultra-fast --duration 30"
echo ""
echo "Para benchmarks largos:"
echo "  python test_fps.py --preset ultra-fast --duration 300"
echo "═══════════════════════════════════════════════════════"
