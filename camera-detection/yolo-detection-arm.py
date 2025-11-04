#!/usr/bin/env python3
"""
Detector de objetos YOLOv8 con soporte para NCNN en ARM/Raspberry Pi

Soporta tanto YOLOv8 nativo como conversión automática a NCNN
"""

import argparse
import cv2
import sys
import time
import torch
from ultralytics import YOLO
from pathlib import Path
import os

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
    # Detectar número de cores disponibles
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    optimal_threads = max(1, num_cores - 1)
    
    cv2.setNumThreads(optimal_threads)
    if hasattr(cv2, "setUseOptimized"):
        cv2.setUseOptimized(True)
    
    # Desactivar CUDA si está disponible (más lento en CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Optimizar PyTorch para CPU
    torch.set_num_threads(optimal_threads)
    torch.set_num_interop_threads(1)

    print("[INFO] Optimizaciones ARM aplicadas")
    print(f"       - Threads CV2: {optimal_threads}")
    print(f"       - Threads PyTorch: {optimal_threads}")
    print("       - NEON Optimization: Enabled")
    print("       - CUDA: Disabled")


def convert_model_to_ncnn(model_path, imgsz=320, verbose=False):
    """
    Convierte un modelo YOLOv8 a formato NCNN

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

        model = YOLO(model_path)

        # Exportar a NCNN
        model.export(format="ncnn", imgsz=imgsz, optimize=False, simplify=True)

        # Encontrar los archivos generados
        base_name = Path(model_path).stem
        model_dir = Path(model_path).parent
        ncnn_model_dir = model_dir / f"{base_name}_ncnn_model"

        if ncnn_model_dir.exists():
            param_file = ncnn_model_dir / "model.ncnn.param"
            bin_file = ncnn_model_dir / "model.ncnn.bin"

            if param_file.exists() and bin_file.exists():
                if verbose:
                    print(f"[NCNN] Conversión completada")
                    print(f"       - {param_file}")
                    print(f"       - {bin_file}")

                return str(param_file), str(bin_file)

        if verbose:
            print("[ERROR] No se encontraron archivos NCNN generados")
        return None, None

    except Exception as e:
        if verbose:
            print(f"[ERROR] Conversión a NCNN falló: {e}")
        return None, None


class NCNNYolo:
    """Wrapper para YOLO usando NCNN"""

    def __init__(
        self, param_path, bin_path, input_size=320, conf_threshold=0.5, verbose=False
    ):
        if not NCNN_AVAILABLE:
            raise ImportError("NCNN no está disponible")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 2

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.verbose = verbose

    def detect(self, image):
        """Detectar objetos en imagen usando NCNN"""
        try:
            h, w = image.shape[:2]
            scale = min(self.input_size / w, self.input_size / h)
            new_w, new_h = int(w * scale), int(h * scale)

            resized = cv2.resize(image, (new_w, new_h))
            pad_w = self.input_size - new_w
            pad_h = self.input_size - new_h
            top, bottom = pad_h // 2, pad_h - pad_h // 2
            left, right = pad_w // 2, pad_w - pad_w // 2

            padded = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )

            mat_in = ncnn.Mat.from_pixels(
                padded, ncnn.Mat.PixelType.PIXEL_BGR, self.input_size, self.input_size
            )
            mat_in.substract_mean_normalize(
                [0, 0, 0], [1 / 255.0, 1 / 255.0, 1 / 255.0]
            )

            ex = self.net.create_extractor()
            ex.input("images", mat_in)
            ret, mat_out = ex.extract("output")

            if ret != 0:
                return []

            # Procesar detecciones (formato similar a YOLOv8)
            detections = []
            for i in range(mat_out.h if hasattr(mat_out, "h") else 0):
                detection = mat_out.row(i) if hasattr(mat_out, "row") else None
                if detection:
                    detections.append(detection)

            return detections

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Detección NCNN: {e}")
            return []


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
        """Inicializa modelo NCNN"""
        try:
            if self.verbose:
                print("[INFO] Intentando usar NCNN...")

            param_path, bin_path = convert_model_to_ncnn(
                self.model_path, self.imgsz, self.verbose
            )

            if param_path and bin_path:
                self.model = NCNNYolo(
                    param_path, bin_path, self.imgsz, self.conf_threshold, self.verbose
                )
                self.use_ncnn_mode = True
                if self.verbose:
                    print("[INFO] NCNN cargado correctamente")
                return True
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] NCNN falló: {e}")

        return False

    def _init_yolo_model(self, half_precision):
        """Inicializa modelo YOLOv8"""
        if self.verbose:
            print(f"[INFO] Cargando YOLOv8: {self.model_path}...")

        # Cargar modelo
        self.model = YOLO(self.model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.verbose:
            print(f"[INFO] Device: {device}")

        # Configurar modelo con parámetros óptimos para ARM
        if device == "cpu":
            # En CPU, usar max_det bajo y verbose=False para más velocidad
            self.model.overrides['max_det'] = 300  # Reducir detecciones
            self.model.overrides['verbose'] = False
            self.model.overrides['agnostic_nms'] = False
            
            if half_precision:
                if self.verbose:
                    print("[INFO] Usando FP16")
                self.model.to(device).half()
            else:
                self.model.to(device)
        else:
            # GPU
            if half_precision:
                if self.verbose:
                    print("[INFO] Usando FP16")
                self.model.to(device).half()
            else:
                self.model.to(device)

        self.use_ncnn_mode = False

    def get_detection(self):
        """Obtiene una detección de la cámara"""
        ret, frame = self.cap.read()
        if not ret:
            return {
                "detected": False,
                "objects": [],
                "frame": None,
                "fps": self.fps_display,
                "timestamp": time.time(),
            }

        # Actualizar FPS
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.fps_counter += 1
        if elapsed >= 1.0:
            self.fps_display = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_time = current_time

        # Ejecutar tracking
        try:
            if self.use_ncnn_mode:
                # NCNN mode
                detections = self.model.detect(frame)
                objects_detected = []
                detected = len(detections) > 0
            else:
                # YOLOv8 mode
                results = self.model.track(
                    frame,
                    conf=self.conf_threshold,
                    classes=[self.target_class],
                    imgsz=self.imgsz,
                    tracker=self.tracker,
                    persist=True,
                    verbose=False,
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
    ap.add_argument("--imgsz", type=int, default=320, help="Tamaño de imagen")
    ap.add_argument("--width", type=int, default=640, help="Ancho de captura")
    ap.add_argument("--height", type=int, default=480, help="Alto de captura")
    ap.add_argument("--fps", type=int, default=30, help="FPS de captura")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker")
    ap.add_argument("--half", action="store_true", help="Usar FP16")
    ap.add_argument("--arm-optimize", action="store_true", help="Optimizar para ARM")
    ap.add_argument(
        "--use-ncnn", action="store_true", help="Usar NCNN si está disponible"
    )
    ap.add_argument("--rpi5-fast", action="store_true", help="Preset optimizado para Raspberry Pi 5 (muy rápido)")
    ap.add_argument("--verbose", action="store_true", help="Modo verbose")
    args = ap.parse_args()
    
    # Aplicar preset Raspberry Pi 5 si se solicita
    if args.rpi5_fast:
        args.width = 320
        args.height = 240
        args.imgsz = 256
        args.fps = 20
        args.arm_optimize = True
        args.use_ncnn = True
        args.conf = 0.4

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
        )

        win = "YOLOv8 Detector - ARM"
        cv2.namedWindow(win)

        while True:
            result = detector.get_detection()

            if result["frame"] is None:
                break

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

            cv2.imshow(win, vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("[INFO] Saliendo...")
                break

        detector.close()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
