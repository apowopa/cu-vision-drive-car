#!/usr/bin/env python3
"""
Script rápido para probar FPS sin mostrar ventanas GUI
Diseñado para Raspberry Pi 5 en headless
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
import time
import argparse
import importlib.util
from pathlib import Path

# Cargar yolo-detection-arm.py dinámicamente (tiene guión en el nombre)
detector_path = Path(__file__).parent / "camera-detection" / "yolo-detection-arm.py"
spec = importlib.util.spec_from_file_location("yolo_detection_arm", detector_path)
if spec and spec.loader:
    yolo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(yolo_module)
    ObjectDetector = yolo_module.ObjectDetector
else:
    print(f"[ERROR] No se pudo cargar yolo-detection-arm.py desde {detector_path}")
    sys.exit(1)

def test_fps(duration=30, **kwargs):
    """Prueba FPS durante N segundos"""
    print("[TEST] Iniciando prueba de FPS...")
    print(f"[TEST] Duración: {duration}s")
    print(f"[TEST] Parámetros: {kwargs}")
    
    detector = ObjectDetector(
        verbose=True,
        **kwargs
    )
    
    print("\n[TEST] Detector inicializado, presione Ctrl+C para detener\n")
    
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    try:
        while time.time() - start_time < duration:
            result = detector.get_detection()
            frame_count += 1
            
            if result["detected"]:
                detection_count += 1
            
            # Mostrar progreso cada segundo
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and frame_count % 30 < 5:
                print(f"[PROGRESO] {int(elapsed)}s - {frame_count} frames - FPS real: {frame_count/elapsed:.1f}")
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrumpido por usuario")
    
    finally:
        detector.close()
    
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*60)
    print("[RESULTADOS]")
    print(f"  Tiempo transcurrido: {elapsed:.1f}s")
    print(f"  Frames procesados: {frame_count}")
    print(f"  Detecciones: {detection_count}")
    print(f"  FPS PROMEDIO: {fps:.1f}")
    print("="*60)
    
    return fps

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test FPS rápido para ARM")
    ap.add_argument("--duration", type=int, default=30, help="Duración en segundos")
    ap.add_argument("--preset", type=str, default="ultra-fast", 
                    help="Preset: ultra-fast (160x120), fast (240x180), balanced (320x240)")
    ap.add_argument("--skip-frames", type=int, default=0, help="Saltear N frames")
    ap.add_argument("--class", type=int, default=0, dest="target_class", help="Clase objetivo")
    ap.add_argument("--conf", type=float, default=0.3, help="Confianza")
    
    args = ap.parse_args()
    
    # Configurar preset
    if args.preset == "ultra-fast":
        kwargs = {
            "width": 160,
            "height": 120,
            "imgsz": 128,
            "fps": 15,
            "arm_optimize": True,
            "skip_frames": args.skip_frames,
            "conf_threshold": args.conf,
            "target_class": args.target_class,
        }
        print("🚀 PRESET: ULTRA-FAST (160x120, imgsz=128)")
    elif args.preset == "fast":
        kwargs = {
            "width": 240,
            "height": 180,
            "imgsz": 160,
            "fps": 20,
            "arm_optimize": True,
            "skip_frames": args.skip_frames,
            "conf_threshold": args.conf,
            "target_class": args.target_class,
        }
        print("🚀 PRESET: FAST (240x180, imgsz=160)")
    elif args.preset == "balanced":
        kwargs = {
            "width": 320,
            "height": 240,
            "imgsz": 192,
            "fps": 20,
            "arm_optimize": True,
            "skip_frames": args.skip_frames,
            "conf_threshold": args.conf,
            "target_class": args.target_class,
        }
        print("🚀 PRESET: BALANCED (320x240, imgsz=192)")
    else:
        print(f"⚠️  Preset desconocido: {args.preset}")
        sys.exit(1)
    
    fps = test_fps(duration=args.duration, **kwargs)
    
    if fps < 15:
        print("⚠️  FPS muy bajo, intente:")
        print("   - python test_fps.py --preset ultra-fast --skip-frames 1")
        print("   - python test_fps.py --preset ultra-fast --skip-frames 2")
    elif fps < 25:
        print("✓ FPS aceptable para tracking (15-25 FPS)")
    else:
        print("✅ FPS excelente (>25 FPS)")
