#!/usr/bin/env python3
"""
Ejemplo avanzado: Controlador de carrito que sigue objetos + evita obstáculos

Combina:
1. Tracker YOLOv8 para seguir objetos
2. Sensores VL53L0X para detección de obstáculos
3. Control automático de motores

Funcionalidad:
- Si hay un objeto a seguir y no hay obstáculos -> lo sigue
- Si hay un objeto pero hay obstáculo muy cerca -> detiene/retrocede
- Si no hay objeto -> se detiene
"""

import time
import sys
import argparse
from pathlib import Path
import os

# Importar el tracker y controlador
script_dir = Path(__file__).parent
project_root = script_dir.parent
camera_detection_dir = project_root / "camera-detection"

# Agregar al path
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    from tracker_controller import TrackerController
except ModuleNotFoundError:
    # Fallback si está en ruta diferente
    sys.path.insert(0, os.path.dirname(__file__))
    from tracker_controller import TrackerController

# Importar sensores
try:
    import busio
    import board
    from gpiozero import DigitalOutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory
    from gpiozero import Device
    from adafruit_vl53l0x import VL53L0X
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False
    print("[WARNING] Librerías de sensores no disponibles")


class SensorController:
    """Controlador de sensores VL53L0X"""
    
    def __init__(self, verbose=False):
        """
        Inicializa los sensores de distancia
        
        Args:
            verbose: Mostrar mensajes de debug
        """
        self.verbose = verbose
        self.sensors = {}
        self.distances = {
            'frontal': 1000,
            'derecha': 1000,
            'izquierda': 1000
        }
        
        if not SENSORS_AVAILABLE:
            print("[WARNING] Sensores no disponibles. Usando valores por defecto.")
            return
        
        try:
            Device.pin_factory = LGPIOFactory()
            
            # MAPEO DE PINES GPIO (XSHUT - Sensor Shutdown)
            # Estos pines controlan el apagado de cada sensor
            # Cuando están en LOW, el sensor se apaga
            # Cuando están en HIGH, el sensor se enciende
            xshut_pins = {
                'frontal': 4,      # GPIO 4  - XSHUT sensor frontal
                'derecha': 17,     # GPIO 17 - XSHUT sensor derecho
                'izquierda': 27    # GPIO 27 - XSHUT sensor izquierdo
            }
            
            # MAPEO DE DIRECCIONES I2C (0x29 es la dirección por defecto)
            # Se pueden cambiar hasta 3 sensores a diferentes direcciones
            # Cada sensor se inicializa con una dirección única
            new_addresses = {
                'izquierda': 0x30,  # Dirección I2C: 0x30 (48 decimal)
                'frontal': 0x31,    # Dirección I2C: 0x31 (49 decimal)
                'derecha': 0x32     # Dirección I2C: 0x32 (50 decimal)
            }
            
            shutdown_pins = {}
            for name, pin in xshut_pins.items():
                shutdown_pins[name] = DigitalOutputDevice(pin, initial_value=False)
            
            # Inicializar I2C
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Configurar cada sensor VL53L0X
            # Proceso de inicialización:
            # 1. Activar sensor con XSHUT HIGH (pin GPIO)
            # 2. Leer del bus I2C (dirección por defecto 0x29)
            # 3. Cambiar dirección I2C a una única por cada sensor
            # 4. Repetir para los otros sensores
            print("[INFO] Inicializando sensores de distancia...")
            print("[INFO] Mapeo de sensores:")
            print("[INFO]   - GPIO 4  → Sensor Frontal  → Dirección I2C: 0x31")
            print("[INFO]   - GPIO 17 → Sensor Derecho  → Dirección I2C: 0x32")
            print("[INFO]   - GPIO 27 → Sensor Izquierdo→ Dirección I2C: 0x30")
            print()
            
            for name, pin_device in shutdown_pins.items():
                pin_device.on()
                time.sleep(0.1)
                
                try:
                    sensor_temp = VL53L0X(i2c)
                    new_addr = new_addresses[name]
                    sensor_temp.set_address(new_addr)
                    self.sensors[name] = sensor_temp
                    print(f"  ✓ Sensor '{name}' inicializado en {hex(new_addr)}")
                except Exception as e:
                    print(f"  ✗ Error con sensor '{name}': {e}")
                    pin_device.off()
                
                time.sleep(0.05)
            
            print("[INFO] Sensores listos!")
            
        except Exception as e:
            print(f"[ERROR] No se pueden inicializar los sensores: {e}")
            print("[INFO] Usando modo sin sensores")
    
    def read_sensors(self):
        """Lee todos los sensores"""
        if not self.sensors:
            return self.distances
        
        for name, sensor in self.sensors.items():
            try:
                self.distances[name] = sensor.range
            except Exception:
                self.distances[name] = 1000  # Valor seguro si falla
        
        return self.distances
    
    def _log(self, message):
        """Log con verbose"""
        if self.verbose:
            print(f"[SENSOR] {message}")


class AdvancedTrackerController(TrackerController):
    """Controlador avanzado que integra tracking + sensores"""
    
    def __init__(self,
                 model_path="yolov8n.pt",
                 camera_idx=0,
                 target_class=0,
                 conf_threshold=0.5,
                 use_ncnn=False,
                 arm_optimize=False,
                 simulation=False,
                 verbose=False,
                 obstacle_distance=200):
        """
        Inicializa el controlador avanzado
        
        Args:
            obstacle_distance: Distancia mínima para obstáculo (mm)
            ... (otros parámetros igual a TrackerController)
        """
        # Inicializar controlador base
        super().__init__(
            model_path=model_path,
            camera_idx=camera_idx,
            target_class=target_class,
            conf_threshold=conf_threshold,
            use_ncnn=use_ncnn,
            arm_optimize=arm_optimize,
            simulation=simulation,
            verbose=verbose
        )
        
        self.obstacle_distance = obstacle_distance
        
        # Inicializar sensores
        print("[INFO] Inicializando sensores...")
        self.sensor_controller = SensorController(verbose=verbose)
        
        print("[INFO] AdvancedTrackerController listo!")
    
    def process_detection(self, result):
        """
        Procesa detección + sensores y controla el carrito
        
        Args:
            result: Dict con resultado de detección
        """
        # Leer sensores
        distances = self.sensor_controller.read_sensors()
        dist_front = distances['frontal']
        dist_left = distances['izquierda']
        dist_right = distances['derecha']
        
        # Mostrar distancias
        self._log(f"Distancias: F={dist_front:4d}mm | I={dist_left:4d}mm | D={dist_right:4d}mm")
        
        # Si hay obstáculo muy cerca, frenar/retroceder
        if dist_front < self.obstacle_distance:
            self._log("¡Obstáculo detectado!")
            self.motor_controller.brake()
            
            # Retroceder después de 1 segundo
            time.sleep(1)
            self.motor_controller.backward()
            time.sleep(0.5)
            self.motor_controller.stop()
            return
        
        # Si hay obstáculo a los lados, evitar
        if dist_left < self.obstacle_distance * 0.8 and dist_right > dist_left:
            self._log("Obstáculo a la izquierda, girando derecha")
            self.motor_controller.turn_right()
            return
        
        if dist_right < self.obstacle_distance * 0.8 and dist_left > dist_right:
            self._log("Obstáculo a la derecha, girando izquierda")
            self.motor_controller.turn_left()
            return
        
        # Sin obstáculos críticos, seguir el objeto
        if not result['detected']:
            self.motor_controller.stop()
            return
        
        objects = result['objects']
        if not objects:
            self.motor_controller.stop()
            return
        
        # Usar el primer objeto
        obj = objects[0]
        position = obj['position']
        confidence = obj['confidence']
        
        self._log(f"Detectado: {position} (confianza: {confidence:.2f})")
        
        # Controlar según posición
        if position == "IZQUIERDA":
            self.motor_controller.turn_left()
        elif position == "DERECHA":
            self.motor_controller.turn_right()
        elif position == "CENTRO":
            self.motor_controller.forward()
    
    def run(self, max_iterations=None, delay=0.1):
        """
        Ejecuta el loop principal
        
        Args:
            max_iterations: Máximo de iteraciones
            delay: Delay entre iteraciones
        """
        iteration = 0
        fps_samples = []
        
        try:
            print("[INFO] Iniciando loop avanzado...")
            print("[INFO] Presiona Ctrl+C para salir")
            print()
            
            while True:
                # Obtener detección
                result = self.detector.get_detection()
                
                # Procesar con sensores
                self.process_detection(result)
                
                # Mostrar información
                distances = self.sensor_controller.distances
                print(f"📏 Distancias: F={distances['frontal']:4d}mm | "
                      f"I={distances['izquierda']:4d}mm | D={distances['derecha']:4d}mm")
                
                if result['detected']:
                    for obj in result['objects']:
                        position = obj['position']
                        confidence = obj['confidence']
                        print(f"✓ {position:10s} | Confianza: {confidence:.2f} | "
                              f"Motor: {self.motor_controller.current_state}")
                else:
                    print(f"✗ Sin detecciones | Motor: {self.motor_controller.current_state}")
                
                # Mostrar FPS
                fps_samples.append(result['fps'])
                if len(fps_samples) >= 10:
                    avg_fps = sum(fps_samples) / len(fps_samples)
                    print(f"📊 FPS promedio: {avg_fps:.1f}")
                    fps_samples = []
                
                print()
                
                # Verificar límite
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
                
                time.sleep(delay)
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupción del usuario")
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[INFO] Deteniendo...")
            self.motor_controller.stop()
            self.motor_controller.cleanup()
            self.detector.close()
            print("[INFO] Finalizado")


def main():
    parser = argparse.ArgumentParser(
        description="Controlador avanzado: sigue objetos y evita obstáculos"
    )
    
    # Tracker
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Modelo YOLOv8")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice de cámara")
    parser.add_argument("--class", type=int, default=0, dest="target_class",
                        help="Clase a detectar (0=personas)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Umbral de confianza")
    
    # Sensores
    parser.add_argument("--obstacle-distance", type=int, default=200,
                        help="Distancia crítica de obstáculo (mm)")
    
    # Optimización
    parser.add_argument("--use-ncnn", action="store_true",
                        help="Usar NCNN")
    parser.add_argument("--arm-optimize", action="store_true",
                        help="Optimizaciones ARM")
    
    # Modo
    parser.add_argument("--simulation", action="store_true",
                        help="Modo simulación")
    parser.add_argument("--verbose", action="store_true",
                        help="Debug info")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Máx iteraciones")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay entre iteraciones")
    
    args = parser.parse_args()
    
    # Crear controlador
    controller = AdvancedTrackerController(
        model_path=args.model,
        camera_idx=args.camera,
        target_class=args.target_class,
        conf_threshold=args.conf,
        use_ncnn=args.use_ncnn,
        arm_optimize=args.arm_optimize,
        simulation=args.simulation,
        verbose=args.verbose,
        obstacle_distance=args.obstacle_distance
    )
    
    # Ejecutar
    controller.run(max_iterations=args.max_iterations, delay=args.delay)


if __name__ == "__main__":
    main()
