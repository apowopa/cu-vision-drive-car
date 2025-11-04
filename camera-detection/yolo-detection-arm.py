#!/usr/bin/env python3
"""
Detector de objetos YOLOv8 con soporte para NCNN en ARM/Raspberry Pi

Soporta tanto YOLOv8 nativo como conversión automática a NCNN
"""

import os
import sys
import argparse
import time
import warnings
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
# Desabilitar GUI COMPLETAMENTE antes de cualquier import de cv2
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Usar xcb en lugar de offscreen (que no existe)
os.environ['DISPLAY'] = ':99'  # Virtual display para xcb
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MPLBACKEND'] = 'Agg'
os.environ["QT_DEBUG_PLUGINS"] = "0"  # Desabilitar debug de Qt
os.environ["QT_XCB_GL_INTEGRATION"] = "none"  # Desabilitar GL integration
os.environ["QT_XCB_SCREEN_LIST"] = ""  # Sin screens
os.environ["QT_QPA_PLATFORMTHEME"] = "minimal"  # Tema minimal

# Suppress Qt warnings antes de importar cv2
warnings.filterwarnings("ignore")

# Importar cv2 
import cv2

# Asegurar que cv2 use el backend correcto
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass

CV2_AVAILABLE = True

# ARM/Raspberry Pi specific imports
try:
    import ncnn

    NCNN_AVAILABLE = True
except ImportError:
    NCNN_AVAILABLE = False


def is_raspberry_pi():
    """Detecta si estamos ejecutando en una Raspberry Pi"""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        return "BCM" in cpuinfo or "Raspberry" in cpuinfo
    except Exception:
        return False


def get_cpu_temperature():
    """Obtiene la temperatura del CPU en Raspberry Pi"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000.0
        return temp
    except Exception:
        return None


def optimize_for_arm():
    """Configuraciones específicas para ARM/Raspberry Pi"""
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    optimal_threads = max(1, num_cores - 1)
    
    # Optimizaciones AGRESIVAS para ARM
    # OpenCV: 1 thread para evitar overhead de sincronización
    cv2.setNumThreads(1)
    if hasattr(cv2, "setUseOptimized"):
        cv2.setUseOptimized(True)
    
    # Desactivar CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # PyTorch optimizaciones
    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('medium')

    print("[INFO] Optimizaciones ARM AGRESIVAS aplicadas")
    print("       - Threads CV2: 1")
    print(f"       - Threads PyTorch: {optimal_threads}")
    print("       - Float32 precision: Medium")


def convert_model_to_ncnn(model_path, imgsz=320, verbose=False):
    """
    Convierte un modelo YOLOv8 a formato NCNN con FP32 precision
    Usa múltiples estrategias para manejar incompatibilidades

    Args:
        model_path: Ruta al modelo YOLOv8 (.pt)
        imgsz: Tamaño de imagen para la conversión
        verbose: Mostrar mensajes detallados

    Returns:
        tuple: (param_path, bin_path) o (None, None) si falla
    """
    try:
        if verbose:
            print(f"[NCNN] Convirtiendo modelo: {model_path}")
            print("[NCNN] FP32 precision (INT8 no soportado en ultralytics)")

        model = YOLO(model_path)

        # Estrategia 1: Exportar con optimize=False y simplify=False
        if verbose:
            print("[NCNN] Intento 1: optimize=False, simplify=False...")
        try:
            model.export(
                format="ncnn",
                imgsz=imgsz,
                optimize=False,
                simplify=False,
            )
        except Exception as e1:
            if verbose:
                print(f"[NCNN] Intento 1 falló: {e1}")
            
            # Estrategia 2: Exportar con simplify=True solamente
            if verbose:
                print("[NCNN] Intento 2: optimize=False, simplify=True...")
            try:
                model.export(
                    format="ncnn",
                    imgsz=imgsz,
                    optimize=False,
                    simplify=True,
                )
            except Exception as e2:
                if verbose:
                    print(f"[NCNN] Intento 2 falló: {e2}")
                
                # Estrategia 3: Exportar sin parámetros opcionales
                if verbose:
                    print("[NCNN] Intento 3: parámetros mínimos...")
                model.export(
                    format="ncnn",
                    imgsz=imgsz,
                )

        # Encontrar los archivos generados
        base_name = Path(model_path).stem
        model_dir = Path(model_path).parent
        ncnn_model_dir = model_dir / f"{base_name}_ncnn_model"

        if ncnn_model_dir.exists():
            param_file = ncnn_model_dir / "model.ncnn.param"
            bin_file = ncnn_model_dir / "model.ncnn.bin"

            if param_file.exists() and bin_file.exists():
                if verbose:
                    print("[NCNN] ✅ Conversión completada")
                    print(f"       Param: {param_file}")
                    print(f"       Bin: {bin_file}")

                return str(param_file), str(bin_file)

        if verbose:
            print("[NCNN] ⚠️ Archivos no encontrados")
        return None, None

    except Exception as e:
        if verbose:
            print(f"[NCNN] ⚠️ Error en conversión: {e}")
        return None, None


class NCNNYolo:
    """Wrapper para YOLO usando NCNN - OPTIMIZADO PARA ARM"""

    def __init__(
        self, param_path, bin_path, input_size=128, conf_threshold=0.3, verbose=False
    ):
        if not NCNN_AVAILABLE:
            raise ImportError("NCNN no está disponible")

        if verbose:
            print("[NCNN] Inicializando red NCNN...")
            print(f"[NCNN] Input size: {input_size}")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        
        # Detectar número óptimo de threads
        import multiprocessing
        optimal_threads = max(1, multiprocessing.cpu_count() - 1)
        self.net.opt.num_threads = optimal_threads
        
        if verbose:
            print(f"[NCNN] Threads: {optimal_threads}")

        try:
            self.net.load_param(param_path)
            self.net.load_model(bin_path)
            if verbose:
                print("[NCNN] ✅ Modelo cargado correctamente")
        except Exception as e:
            if verbose:
                print(f"[NCNN] ❌ Error cargando modelo: {e}")
            raise

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.verbose = verbose
        self.track_id_counter = 0
        
        # Cache layer names para evitar búsqueda repetida
        self.input_layer_name = None
        self.output_layer_name = None

    def track(self, image, classes=None, **kwargs):
        """Detectar y trackear objetos (interfaz compatible con YOLOv8)"""
        try:
            h, w = image.shape[:2]
            
            # Preparar imagen
            img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalizar
            mat_in = ncnn.Mat.from_pixels(
                img, ncnn.Mat.PixelType.PIXEL_RGB, self.input_size, self.input_size
            )
            
            # Normalizar a [0, 1]
            mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
            
            # Inferencia - suprimir warnings de NCNN
            import contextlib
            import io
            
            # Silenciar stderr de NCNN
            f_err = io.StringIO()
            with contextlib.redirect_stderr(f_err):
                ex = self.net.create_extractor()
                
                # Intentar con nombres comunes de input/output
                # Usar cache si ya se encontraron los layer names
                if self.input_layer_name is None or self.output_layer_name is None:
                    input_names = ["images", "in0", "input"]
                    output_names = ["output", "out0", "output0"]
                    
                    ret = -1
                    for input_name in input_names:
                        ret = ex.input(input_name, mat_in)
                        if ret == 0:
                            self.input_layer_name = input_name
                            if self.verbose:
                                print(f"[NCNN] ✓ Input layer encontrado: {input_name}")
                            break
                    
                    if ret != 0:
                        if self.verbose:
                            print("[NCNN] ⚠️ Error: no se encontró layer de input")
                        return Results(boxes=[], conf=[], cls=[], ids=None)
                    
                    # Intentar extraer output con nombres comunes
                    mat_out = None
                    for output_name in output_names:
                        ret, mat_out = ex.extract(output_name)
                        if ret == 0:
                            self.output_layer_name = output_name
                            if self.verbose:
                                print(f"[NCNN] ✓ Output layer encontrado: {output_name}")
                            break
                    
                    if ret != 0 or mat_out is None:
                        if self.verbose:
                            print("[NCNN] ⚠️ Error: no se encontró layer de output")
                        return Results(boxes=[], conf=[], cls=[], ids=None)
                else:
                    # Usar layer names cacheados
                    ret = ex.input(self.input_layer_name, mat_in)
                    if ret != 0:
                        return Results(boxes=[], conf=[], cls=[], ids=None)
                    
                    ret, mat_out = ex.extract(self.output_layer_name)
                    if ret != 0 or mat_out is None:
                        return Results(boxes=[], conf=[], cls=[], ids=None)
            
            # Parsear detecciones (formato: x, y, w, h, conf, class_probs...)
            detections = []
            h_mat = mat_out.h
            
            # Iterar sobre detecciones
            for i in range(h_mat):
                data = mat_out.row(i)
                if len(data) >= 6:
                    x_c, y_c, box_w, box_h, conf = data[:5]
                    
                    if conf >= self.conf_threshold:
                        # Convertir a bbox
                        x1 = int((x_c - box_w/2) * w / self.input_size)
                        y1 = int((y_c - box_h/2) * h / self.input_size)
                        x2 = int((x_c + box_w/2) * w / self.input_size)
                        y2 = int((y_c + box_h/2) * h / self.input_size)
                        
                        # Clase
                        cls_idx = 0
                        if len(data) > 5:
                            cls_probs = data[5:]
                            cls_idx = int(np.argmax(cls_probs))
                        
                        # Filtrar por clase si se especifica
                        if classes is not None:
                            if cls_idx not in classes:
                                continue
                        
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'conf': float(conf),
                            'cls': cls_idx,
                            'id': self.track_id_counter
                        })
                        self.track_id_counter += 1
            
            # Convertir a formato YOLOv8
            if detections:
                boxes = np.array([d['box'] for d in detections], dtype=np.float32)
                confs = np.array([d['conf'] for d in detections], dtype=np.float32)
                clss = np.array([d['cls'] for d in detections], dtype=np.int32)
                ids = np.array([d['id'] for d in detections], dtype=np.int32)
                
                return Results(
                    boxes=torch.tensor(boxes),
                    conf=torch.tensor(confs),
                    cls=torch.tensor(clss),
                    ids=torch.tensor(ids)
                )
            else:
                return Results(boxes=[], conf=[], cls=[], ids=None)

        except Exception as e:
            if self.verbose:
                print(f"[NCNN] ❌ Error en track: {e}")
                import traceback
                traceback.print_exc()
            return Results(boxes=[], conf=[], cls=[], ids=None)
            if self.verbose:
                print(f"[WARNING] Detección NCNN: {e}")
            return Results(boxes=[], conf=[], cls=[], ids=None)


class Results:
    """Clase compatible con resultados de YOLO"""
    def __init__(self, boxes, conf, cls, ids):
        self.boxes = boxes if isinstance(boxes, torch.Tensor) else torch.tensor(boxes if boxes else [])
        self.conf = conf if isinstance(conf, torch.Tensor) else torch.tensor(conf if conf else [])
        self.cls = cls if isinstance(cls, torch.Tensor) else torch.tensor(cls if cls else [])
        self.id = ids


class ObjectDetector:
    """Detector de objetos con tracking - Soporta YOLOv8 y NCNN"""

    def __init__(
        self,
        model_path="yolov8n.pt",
        camera_idx=0,
        conf_threshold=0.2,
        target_class=0,
        imgsz=320,
        width=640,
        height=480,
        fps=30,
        tracker="bytetrack.yaml",
        half_precision=False,
        arm_optimize=False,
        use_ncnn=False,
        verbose=False,
        skip_frames=0,  # NUEVO: saltar N frames
    ):
        self.model_path = model_path
        self.camera_idx = camera_idx
        self.conf_threshold = conf_threshold
        self.target_class = target_class
        self.imgsz = imgsz
        self.width = width
        self.height = height
        self.fps = fps
        self.tracker = tracker
        self.verbose = verbose
        self.use_ncnn_mode = False
        self.model = None
        self.skip_frames = skip_frames
        self.frame_count = 0  # NUEVO: contador de frames

        # Detectar si estamos en Raspberry Pi
        self.on_pi = is_raspberry_pi()
        if self.on_pi and self.verbose:
            print("[INFO] Raspberry Pi detectada")

        # Aplicar optimizaciones ARM
        if arm_optimize or self.on_pi:
            optimize_for_arm()
            if self.on_pi and self.verbose:
                temp = get_cpu_temperature()
                if temp:
                    print(f"[INFO] Temperatura CPU: {temp:.1f}°C")

        # Intentar usar NCNN si está disponible
        if use_ncnn and NCNN_AVAILABLE:
            self._init_ncnn_model()

        # Si NCNN no funcionó, usar YOLOv8
        if self.model is None:
            self._init_yolo_model(half_precision)

        # Abrir cámara
        if self.verbose:
            print(f"[INFO] Abriendo cámara {camera_idx}...")
        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara: {camera_idx}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.eff_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.eff_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.eff_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.verbose:
            print(f"[INFO] Captura: {self.eff_w}x{self.eff_h} @ {self.eff_fps:.1f}FPS")

        self.division_line1_x = self.eff_w // 4
        self.division_line2_x = (self.eff_w * 3) // 4

        self.last_time = time.time()
        self.fps_counter = 0
        self.fps_display = 0

    def _init_ncnn_model(self):
        """Inicializa modelo NCNN - OPTIMIZADO PARA ARM"""
        if not NCNN_AVAILABLE:
            if self.verbose:
                print("[NCNN] ⚠️ NCNN no está instalado, usando YOLOv8")
            return False
        
        try:
            if self.verbose:
                print("[NCNN] Intentando inicializar NCNN...")

            param_path, bin_path = convert_model_to_ncnn(
                self.model_path, self.imgsz, self.verbose
            )

            if param_path and bin_path:
                if self.verbose:
                    print("[NCNN] Cargando modelo NCNN...")
                
                self.model = NCNNYolo(
                    param_path=param_path,
                    bin_path=bin_path,
                    input_size=self.imgsz,
                    conf_threshold=self.conf_threshold,
                    verbose=self.verbose
                )
                self.use_ncnn_mode = True
                if self.verbose:
                    print("[NCNN] ✅ Modelo NCNN listo para usar")
                return True
            else:
                if self.verbose:
                    print("[NCNN] ⚠️ Conversión falló, usando YOLOv8")
                return False
                
        except Exception as e:
            if self.verbose:
                print(f"[NCNN] ⚠️ Error inicializando NCNN: {e}")
                print("[NCNN] Fallback a YOLOv8...")
            return False

    def _init_yolo_model(self, half_precision):
        """Inicializa modelo YOLOv8 - OPTIMIZADO MÁXIMO PARA ARM"""
        if self.verbose:
            print(f"[INFO] Cargando YOLOv8: {self.model_path}...")

        # Cargar modelo
        self.model = YOLO(self.model_path)
        
        device = "cpu"  # SIEMPRE CPU en ARM
        if self.verbose:
            print(f"[INFO] Device: {device}")

        # CONFIGURAR CON PARÁMETROS AGRESIVOS PARA ARM
        self.model.overrides['max_det'] = 100  # MÁS REDUCIDO
        self.model.overrides['verbose'] = False
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['amp'] = False  # Deshabilitar automatic mixed precision
        self.model.overrides['device'] = 0  # Forzar CPU
        
        # Desabilitar postprocesamiento pesado
        self.model.overrides['iou'] = 0.45  # Lower IOU para menos NMS
        
        self.model.to(device)
        self.use_ncnn_mode = False

    def get_detection(self):
        """Obtiene una detección de la cámara - OPTIMIZADO PARA ARM"""
        ret, frame = self.cap.read()
        if not ret:
            return {
                "detected": False,
                "objects": [],
                "frame": None,
                "fps": self.fps_display,
                "timestamp": time.time(),
            }

        # Contar frames para saltar
        self.frame_count += 1
        
        # Si debemos saltar este frame, retornar cache
        if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
            return {
                "detected": False,
                "objects": [],
                "frame": frame,
                "fps": self.fps_display,
                "timestamp": time.time(),
            }

        # Reducir resolución drasticamente para ARM (160x120 = 1/16 de píxeles)
        frame_small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_LINEAR)

        # Actualizar FPS
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.fps_counter += 1
        if elapsed >= 1.0:
            self.fps_display = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_time = current_time

        # Ejecutar tracking - SOPORTE PARA NCNN Y YOLOV8
        try:
            # Seleccionar backend
            if self.use_ncnn_mode:
                # NCNN ya redimensiona internamente
                results = self.model.track(
                    frame,  # NCNN maneja resize internamente
                    classes=[self.target_class],
                    imgsz=self.imgsz,
                )
            else:
                # YOLOv8 nativo
                frame_small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_LINEAR)
                results = self.model.track(
                    frame_small,
                    conf=self.conf_threshold,
                    classes=[self.target_class],
                    imgsz=128,
                    tracker=self.tracker,
                    persist=True,
                    verbose=False,
                    device='cpu',
                    half=False,
                    augment=False,
                )
            
            r = results[0] if isinstance(results, list) else results

            objects_detected = []
            detected = False

            if r.boxes is not None and len(r.boxes) > 0:
                detected = True

                boxes = r.boxes.xyxy
                if hasattr(boxes, "cpu"):
                    boxes = boxes.cpu().numpy()
                boxes = boxes.astype(int)

                confs = r.boxes.conf
                if hasattr(confs, "cpu"):
                    confs = confs.cpu().numpy()

                has_track_ids = False
                track_ids = None
                if hasattr(r.boxes, "id") and r.boxes.id is not None:
                    track_ids = r.boxes.id
                    if hasattr(track_ids, "cpu"):
                        track_ids = track_ids.cpu().numpy()
                    if len(track_ids) > 0:
                        track_ids = track_ids.astype(int)
                        has_track_ids = True

                for i, box in enumerate(boxes):
                    try:
                        x1, y1, x2, y2 = box
                        conf = float(confs[i])

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if center_x < self.division_line1_x:
                            position = "IZQUIERDA"
                        elif center_x < self.division_line2_x:
                            position = "CENTRO"
                        else:
                            position = "DERECHA"

                        track_id = None
                        if has_track_ids and track_ids is not None:
                            track_id = int(track_ids[i])

                        obj_info = {
                            "class": 0,
                            "position": position,
                            "confidence": conf,
                            "track_id": track_id,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "center": (int(center_x), int(center_y)),
                        }
                        objects_detected.append(obj_info)

                    except Exception as e:
                        if self.verbose:
                            print(f"[ERROR] Procesando detección {i}: {e}")

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Tracking: {e}")
            return {
                "detected": False,
                "objects": [],
                "frame": frame,
                "fps": self.fps_display,
                "timestamp": time.time(),
            }

        return {
            "detected": detected,
            "objects": objects_detected,
            "frame": frame,
            "fps": self.fps_display,
            "timestamp": time.time(),
        }

    def close(self):
        """Liberar recursos"""
        if self.cap:
            self.cap.release()
        if self.verbose:
            print("[INFO] Detector cerrado")


def main():
    """Script principal"""
    ap = argparse.ArgumentParser(
        description="YOLOv8 Detector con soporte NCNN para ARM"
    )
    ap.add_argument("--camera", type=int, default=0, help="Índice de cámara")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo YOLOv8")
    ap.add_argument("--conf", type=float, default=0.2, help="Confianza mínima")
    ap.add_argument("--class", type=int, default=0, dest="target_class", help="Clase objetivo (0=person, 1=bicycle, etc)")
    ap.add_argument("--imgsz", type=int, default=128, help="Tamaño de imagen")
    ap.add_argument("--width", type=int, default=160, help="Ancho de captura")
    ap.add_argument("--height", type=int, default=120, help="Alto de captura")
    ap.add_argument("--fps", type=int, default=15, help="FPS de captura")
    ap.add_argument("--skip-frames", type=int, default=0, help="Saltar N frames entre detecciones")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker")
    ap.add_argument("--half", action="store_true", help="Usar FP16")
    ap.add_argument("--arm-optimize", action="store_true", help="Optimizar para ARM")
    ap.add_argument(
        "--use-ncnn", action="store_true", help="Usar NCNN SI O SI para máxima velocidad"
    )
    ap.add_argument("--rpi5-ultra-fast", action="store_true", help="MÁXIMA OPTIMIZACIÓN para Raspberry Pi 5 (160x120, imgsz=128)")
    ap.add_argument("--verbose", action="store_true", help="Modo verbose")
    args = ap.parse_args()
    
    # Aplicar preset ULTRA RÁPIDO si se solicita
    if args.rpi5_ultra_fast:
        args.width = 160
        args.height = 120
        args.imgsz = 128
        args.fps = 15
        args.skip_frames = 0
        args.arm_optimize = True
        args.use_ncnn = True  # NCNN HABILITADO para máxima velocidad
        args.conf = 0.3

    try:
        detector = ObjectDetector(
            model_path=args.model,
            camera_idx=args.camera,
            conf_threshold=args.conf,
            target_class=args.target_class,
            imgsz=args.imgsz,
            width=args.width,
            height=args.height,
            fps=args.fps,
            tracker=args.tracker,
            half_precision=args.half,
            arm_optimize=args.arm_optimize,
            use_ncnn=args.use_ncnn,
            verbose=args.verbose,
            skip_frames=args.skip_frames,
        )

        # Intentar crear ventana solo si hay display disponible
        win = "YOLOv8 Detector - ARM"
        has_display = False
        try:
            # Intentar crear ventana de OpenCV
            # Esto puede fallar en headless si Qt no está disponible
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            has_display = True
            if detector.verbose:
                print("[INFO] Display available - visualization ENABLED")
        except Exception as e:
            # En headless o sin display, esto falla silenciosamente
            has_display = False
            if detector.verbose:
                print(f"[INFO] No display available (headless mode): {type(e).__name__}: {e}")
                print("[INFO] Running detection WITHOUT visualization")

        while True:
            result = detector.get_detection()

            if result["frame"] is None:
                break

            # Solo procesar visualización si hay display
            if not has_display:
                continue

            vis_frame = result["frame"].copy()

            # Dibujar líneas de división
            cv2.line(
                vis_frame,
                (detector.division_line1_x, 0),
                (detector.division_line1_x, detector.eff_h),
                (255, 255, 0),
                2,
            )
            cv2.line(
                vis_frame,
                (detector.division_line2_x, 0),
                (detector.division_line2_x, detector.eff_h),
                (255, 255, 0),
                2,
            )

            # Mostrar zonas
            cv2.putText(
                vis_frame,
                "IZQUIERDA",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                vis_frame,
                "CENTRO",
                (detector.eff_w // 2 - 50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis_frame,
                "DERECHA",
                (detector.eff_w - 120, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # Dibujar detecciones
            if result["detected"]:
                for obj in result["objects"]:
                    x1, y1, x2, y2 = obj["bbox"]
                    cx, cy = obj["center"]
                    position = obj["position"]
                    conf = obj["confidence"]
                    track_id = obj["track_id"]

                    if position == "IZQUIERDA":
                        color = (0, 0, 255)
                    elif position == "CENTRO":
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)

                    cv2.circle(vis_frame, (cx, cy), 5, color, -1)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                    if track_id is not None:
                        label = f"ID:{track_id} {position} ({conf:.2f})"
                    else:
                        label = f"{position} ({conf:.2f})"

                    cv2.putText(
                        vis_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # Mostrar información
            info_text = f"FPS: {result['fps']:.1f} | Res: {detector.eff_w}x{detector.eff_h} | Mode: {'NCNN' if detector.use_ncnn_mode else 'YOLOv8'}"
            cv2.putText(
                vis_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            if has_display:
                cv2.imshow(win, vis_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("[INFO] Saliendo...")
                    break

        detector.close()
        if has_display:
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
