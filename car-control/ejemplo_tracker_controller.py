#!/usr/bin/env python3
"""
Ejemplo simple de uso del tracker_controller

Este script demuestra cómo usar el controlador básico sin sensores.
Perfecto para pruebas iniciales.
"""

from tracker_controller import TrackerController


def ejemplo_1_simulacion():
    """EJEMPLO 1: Prueba en modo simulación (sin hardware)"""
    print("\n" + "="*60)
    print("EJEMPLO 1: Modo Simulación")
    print("="*60)
    print("El carrito simula movimientos en consola sin hardware real")
    print()
    
    controller = TrackerController(
        model_path="yolov8n.pt",
        camera_idx=0,
        target_class=0,  # Personas
        simulation=True,
        verbose=True
    )
    
    # Ejecutar por 50 iteraciones
    controller.run(max_iterations=50, delay=0.1)


def ejemplo_2_seguimiento_personas():
    """EJEMPLO 2: Seguimiento de personas (hardware real)"""
    print("\n" + "="*60)
    print("EJEMPLO 2: Seguimiento de Personas (Hardware Real)")
    print("="*60)
    print("El carrito seguirá personas detectadas")
    print()
    
    controller = TrackerController(
        model_path="yolov8n.pt",
        camera_idx=0,
        target_class=0,  # Personas
        conf_threshold=0.5,
        arm_optimize=True,
        simulation=False,
        verbose=True
    )
    
    controller.run(max_iterations=None, delay=0.1)


def ejemplo_3_seguimiento_autos():
    """EJEMPLO 3: Seguimiento de autos (hardware real)"""
    print("\n" + "="*60)
    print("EJEMPLO 3: Seguimiento de Autos (Hardware Real)")
    print("="*60)
    print("El carrito seguirá autos detectados")
    print()
    
    controller = TrackerController(
        model_path="yolov8n.pt",
        camera_idx=0,
        target_class=2,  # Cars
        conf_threshold=0.5,
        arm_optimize=True,
        use_ncnn=True,  # Usar NCNN para más velocidad
        simulation=False,
        verbose=False  # Sin verbose para menos output
    )
    
    controller.run(max_iterations=None, delay=0.1)


def ejemplo_4_raspberry_pi_optimizado():
    """EJEMPLO 4: Optimizado para Raspberry Pi"""
    print("\n" + "="*60)
    print("EJEMPLO 4: Optimizado para Raspberry Pi")
    print("="*60)
    print("Máximas optimizaciones para Raspberry Pi 4")
    print()
    
    controller = TrackerController(
        model_path="yolov8n.pt",  # Nano para velocidad
        camera_idx=0,
        target_class=0,  # Personas
        conf_threshold=0.6,
        arm_optimize=True,
        use_ncnn=True,  # Crucial para Raspberry Pi
        simulation=False,
        verbose=True
    )
    
    controller.run(max_iterations=None, delay=0.15)


def ejemplo_5_prueba_rapida():
    """EJEMPLO 5: Prueba rápida de 10 iteraciones"""
    print("\n" + "="*60)
    print("EJEMPLO 5: Prueba Rápida (10 iteraciones)")
    print("="*60)
    print()
    
    controller = TrackerController(
        model_path="yolov8n.pt",
        camera_idx=0,
        simulation=True,  # Simulación para no necesitar hardware
        verbose=False
    )
    
    controller.run(max_iterations=10, delay=0.1)
    print("\n✓ Prueba completada!")


def menu_principal():
    """Menú principal interactivo"""
    while True:
        print("\n" + "="*60)
        print("TRACKER CONTROLLER - EJEMPLOS")
        print("="*60)
        print("1. Simulación (sin hardware)")
        print("2. Seguimiento de personas (hardware real)")
        print("3. Seguimiento de autos (hardware real + NCNN)")
        print("4. Optimizado para Raspberry Pi")
        print("5. Prueba rápida")
        print("6. Salir")
        print()
        
        opcion = input("Selecciona una opción (1-6): ").strip()
        
        try:
            if opcion == "1":
                ejemplo_1_simulacion()
            elif opcion == "2":
                ejemplo_2_seguimiento_personas()
            elif opcion == "3":
                ejemplo_3_seguimiento_autos()
            elif opcion == "4":
                ejemplo_4_raspberry_pi_optimizado()
            elif opcion == "5":
                ejemplo_5_prueba_rapida()
            elif opcion == "6":
                print("\n¡Hasta luego!")
                break
            else:
                print("✗ Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\n✗ Interrumpido por el usuario")
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    import sys
    
    # Si hay argumentos, ejecutar directamente
    if len(sys.argv) > 1:
        ejemplo = sys.argv[1]
        
        if ejemplo == "1":
            ejemplo_1_simulacion()
        elif ejemplo == "2":
            ejemplo_2_seguimiento_personas()
        elif ejemplo == "3":
            ejemplo_3_seguimiento_autos()
        elif ejemplo == "4":
            ejemplo_4_raspberry_pi_optimizado()
        elif ejemplo == "5":
            ejemplo_5_prueba_rapida()
        else:
            print(f"✗ Ejemplo desconocido: {ejemplo}")
            print("Disponibles: 1, 2, 3, 4, 5")
    else:
        # Sino, mostrar menú interactivo
        menu_principal()
