#!/usr/bin/env python3
"""
Controlador de carrito que sigue objetos detectados por el tracker YOLOv8

El carrito gira hacia la posición del objeto detectado:
- IZQUIERDA: gira a la izquierda
- CENTRO: avanza recto
- DERECHA: gira a la derecha
- Sin detección: se detiene

Requiere:
- gpiozero y su factory LGPIO (para control de motores)
- El script del tracker: yolo-detection-arm.py
"""

import time
import sys
import argparse
from pathlib import Path
import os

# Importar el tracker
script_dir = Path(__file__).parent
project_root = script_dir.parent
camera_detection_dir = project_root / "camera-detection"

# Agregar al path
if str(camera_detection_dir) not in sys.path:
    sys.path.insert(0, str(camera_detection_dir))

try:
    from yolo_detection_arm import ObjectDetector
except ModuleNotFoundError:
    # Fallback si está en ruta diferente
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'camera-detection'))
    from yolo_detection_arm import ObjectDetector

# Importar gpiozero para control de motores
try:
    from gpiozero import DigitalOutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False
    print("[WARNING] gpiozero no disponible. Modo simulación.")


class MotorController:
    """Controlador de motores para el carrito"""
    
    def __init__(self, use_simulation=False, verbose=False):
        """
        Inicializa el controlador de motores
        
        Args:
            use_simulation: Si True, simula los movimientos en consola
            verbose: Mostrar mensajes de debug
        """
        self.use_simulation = use_simulation
        self.verbose = verbose
        self.current_state = "STOP"
        
        if not use_simulation and not GPIOZERO_AVAILABLE:
            print("[ERROR] gpiozero no disponible. Cambiando a modo simulación.")
            self.use_simulation = True
        
        if not self.use_simulation:
            try:
                # Configurar pin factory
                Device.pin_factory = LGPIOFactory()
                
                # Configurar pines de los motores
                self.motor_left_a = DigitalOutputDevice(5)
                self.motor_left_b = DigitalOutputDevice(6)
                self.motor_right_a = DigitalOutputDevice(13)
                self.motor_right_b = DigitalOutputDevice(19)
                
                print("[INFO] Motores inicializados correctamente")
            except Exception as e:
                print(f"[ERROR] No se pueden inicializar los motores: {e}")
                print("[INFO] Cambiando a modo simulación.")
                self.use_simulation = True
    
    def _log(self, message):
        """Log con verbose"""
        if self.verbose:
            print(f"[MOTOR] {message}")
    
    def brake(self):
        """Frena el carrito (todos los motores ON)"""
        if self.current_state == "BRAKE":
            return
        
        self._log("BRAKE")
        self.current_state = "BRAKE"
        
        if not self.use_simulation:
            self.motor_left_a.on()
            self.motor_left_b.on()
            self.motor_right_a.on()
            self.motor_right_b.on()
        else:
            print("🛑 BRAKE")
    
    def stop(self):
        """Detiene el carrito (todos los motores OFF)"""
        if self.current_state == "STOP":
            return
        
        self._log("STOP")
        self.current_state = "STOP"
        
        if not self.use_simulation:
            self.motor_left_a.off()
            self.motor_left_b.off()
            self.motor_right_a.off()
            self.motor_right_b.off()
        else:
            print("⏹️  STOP")
    
    def forward(self):
        """Avanza recto"""
        if self.current_state == "FORWARD":
            return
        
        self._log("FORWARD")
        self.current_state = "FORWARD"
        
        if not self.use_simulation:
            self.motor_left_a.on()
            self.motor_left_b.off()
            self.motor_right_a.on()
            self.motor_right_b.off()
        else:
            print("⬆️  FORWARD")
    
    def backward(self):
        """Retrocede"""
        if self.current_state == "BACKWARD":
            return
        
        self._log("BACKWARD")
        self.current_state = "BACKWARD"
        
        if not self.use_simulation:
            self.motor_left_a.off()
            self.motor_left_b.on()
            self.motor_right_a.off()
            self.motor_right_b.on()
        else:
            print("⬇️  BACKWARD")
    
    def turn_left(self):
        """Gira a la izquierda"""
        if self.current_state == "LEFT":
            return
        
        self._log("LEFT")
        self.current_state = "LEFT"
        
        if not self.use_simulation:
            self.motor_left_a.off()
            self.motor_left_b.on()
            self.motor_right_a.on()
            self.motor_right_b.off()
        else:
            print("⬅️  LEFT")
    
    def turn_right(self):
        """Gira a la derecha"""
        if self.current_state == "RIGHT":
            return
        
        self._log("RIGHT")
        self.current_state = "RIGHT"
        
        if not self.use_simulation:
            self.motor_left_a.on()
            self.motor_left_b.off()
            self.motor_right_a.off()
            self.motor_right_b.on()
        else:
            print("➡️  RIGHT")
    
    def cleanup(self):
        """Limpia los pines"""
        if not self.use_simulation:
            try:
                self.stop()
                self.motor_left_a.close()
                self.motor_left_b.close()
                self.motor_right_a.close()
                self.motor_right_b.close()
            except Exception as e:
                print(f"[ERROR] Error al limpiar los pines: {e}")


class TrackerController:
    """Controlador que integra el tracker YOLOv8 con el control del carrito"""
    
    def __init__(self, 
                 model_path="yolov8n.pt",
                 camera_idx=0,
                 target_class=0,
                 conf_threshold=0.5,
                 use_ncnn=False,
                 arm_optimize=False,
                 simulation=False,
                 verbose=False):
        """
        Inicializa el controlador de seguimiento
        
        Args:
            model_path: Ruta al modelo YOLOv8
            camera_idx: Índice de la cámara
            target_class: Clase objetivo a detectar (0=person)
            conf_threshold: Umbral de confianza (0.0-1.0)
            use_ncnn: Usar NCNN si está disponible
            arm_optimize: Aplicar optimizaciones ARM
            simulation: Modo simulación (sin hardware)
            verbose: Mostrar mensajes de debug
        """
        self.verbose = verbose
        
        print("[INFO] Inicializando TrackerController...")
        
        # Inicializar detector
        print("[INFO] Inicializando detector YOLOv8...")
        self.detector = ObjectDetector(
            model_path=model_path,
            camera_idx=camera_idx,
            conf_threshold=conf_threshold,
            target_class=target_class,
            imgsz=320,
            width=640,
            height=480,
            fps=30,
            tracker="bytetrack.yaml",
            arm_optimize=arm_optimize,
            use_ncnn=use_ncnn,
            verbose=verbose
        )
        
        # Inicializar controlador de motores
        print("[INFO] Inicializando controlador de motores...")
        self.motor_controller = MotorController(
            use_simulation=simulation,
            verbose=verbose
        )
        
        print("[INFO] TrackerController listo!")
    
    def _log(self, message):
        """Log con verbose"""
        if self.verbose:
            print(f"[TRACKER] {message}")
    
    def process_detection(self, result):
        """
        Procesa la detección y controla el carrito
        
        Args:
            result: Dict con el resultado de la detección
        """
        if not result['detected']:
            # Sin objetos detectados
            self.motor_controller.stop()
            return
        
        # Obtener el primer objeto detectado (o el más cercano)
        objects = result['objects']
        if not objects:
            self.motor_controller.stop()
            return
        
        # Usar el primer objeto (podrías cambiar la lógica aquí)
        obj = objects[0]
        position = obj['position']
        confidence = obj['confidence']
        
        self._log(f"Detectado: {position} (confianza: {confidence:.2f})")
        
        # Controlar el carrito según la posición
        if position == "IZQUIERDA":
            self.motor_controller.turn_left()
        elif position == "DERECHA":
            self.motor_controller.turn_right()
        elif position == "CENTRO":
            self.motor_controller.forward()
    
    def run(self, max_iterations=None, delay=0.1):
        """
        Ejecuta el loop principal de seguimiento
        
        Args:
            max_iterations: Máximo número de iteraciones (None = infinito)
            delay: Delay entre iteraciones en segundos
        """
        iteration = 0
        fps_samples = []
        
        try:
            print("[INFO] Iniciando loop de seguimiento...")
            print("[INFO] Presiona Ctrl+C para salir")
            print()
            
            while True:
                # Obtener detección
                result = self.detector.get_detection()
                
                # Procesar detección y controlar carrito
                self.process_detection(result)
                
                # Mostrar información
                if result['detected']:
                    for obj in result['objects']:
                        position = obj['position']
                        confidence = obj['confidence']
                        print(f"✓ {position:10s} | Confianza: {confidence:.2f}")
                else:
                    print("✗ Sin detecciones")
                
                # Mostrar FPS
                fps_samples.append(result['fps'])
                if len(fps_samples) >= 10:
                    avg_fps = sum(fps_samples) / len(fps_samples)
                    print(f"   FPS promedio: {avg_fps:.1f}")
                    fps_samples = []
                
                # Verificar límite de iteraciones
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Delay
                time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupción del usuario")
        except Exception as e:
            print(f"\n[ERROR] Error durante la ejecución: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[INFO] Deteniendo...")
            self.motor_controller.stop()
            self.motor_controller.cleanup()
            self.detector.close()
            print("[INFO] Finalizado")
    
    def cleanup(self):
        """Limpia recursos"""
        self.motor_controller.cleanup()
        self.detector.close()


def main():
    parser = argparse.ArgumentParser(
        description="Controlador de carrito que sigue objetos detectados por YOLOv8"
    )
    
    # Argumentos del detector
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Ruta al modelo YOLOv8 (default: yolov8n.pt)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice de la cámara (default: 0)")
    parser.add_argument("--class", type=int, default=0, dest="target_class",
                        help="Clase objetivo a detectar (default: 0 = personas)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Umbral de confianza (default: 0.5)")
    
    # Opciones de optimización
    parser.add_argument("--use-ncnn", action="store_true",
                        help="Usar NCNN si está disponible")
    parser.add_argument("--arm-optimize", action="store_true",
                        help="Aplicar optimizaciones para ARM/Raspberry Pi")
    
    # Modo de ejecución
    parser.add_argument("--simulation", action="store_true",
                        help="Modo simulación (sin hardware)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar mensajes de debug")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Máximo número de iteraciones (default: infinito)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay entre iteraciones en segundos (default: 0.1)")
    
    args = parser.parse_args()
    
    # Crear controlador
    controller = TrackerController(
        model_path=args.model,
        camera_idx=args.camera,
        target_class=args.target_class,
        conf_threshold=args.conf,
        use_ncnn=args.use_ncnn,
        arm_optimize=args.arm_optimize,
        simulation=args.simulation,
        verbose=args.verbose
    )
    
    # Ejecutar
    controller.run(max_iterations=args.max_iterations, delay=args.delay)


if __name__ == "__main__":
    main()
