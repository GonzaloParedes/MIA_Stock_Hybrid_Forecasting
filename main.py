import sys
import os
import subprocess
import time

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
# Ajusta estos nombres si tus carpetas o archivos se llaman diferente

CONFIG = {
    "A": {
        "folder": "VarianteA",         # Nombre de la carpeta de la Variante A
        "script": "orchestrator.py",   # Nombre del script principal de A (cámbialo si es main.py, etc.)
        "desc": "Arquitectura Variante A (Orquestador Original)"
    },
    "B": {
        "folder": "varianteB",         # Nombre de la carpeta de la Variante B
        "script": "replicate_paper.py",# Nombre del script principal de B
        "desc": "Arquitectura Variante B (True Batch / Paper Replication)"
    }
}

# ==============================================================================
# LÓGICA DE EJECUCIÓN
# ==============================================================================

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_variant(option):
    cfg = CONFIG[option]
    folder_path = os.path.join(os.getcwd(), cfg["folder"])
    script_path = os.path.join(folder_path, cfg["script"])

    # 1. Verificaciones de seguridad
    if not os.path.exists(folder_path):
        print(f"\nERROR: No encuentro la carpeta: {folder_path}")
        print(f"Asegúrate de que la carpeta '{cfg['folder']}' esté en el mismo lugar que este script.")
        return

    if not os.path.exists(script_path):
        print(f"\nERROR: No encuentro el script: {cfg['script']}")
        print(f"Buscado en: {script_path}")
        return

    # 2. Ejecución
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {cfg['desc']}")
    print(f"Directorio de trabajo: {folder_path}")
    print(f"{'='*60}\n")

    try:
        # Usamos subprocess para ejecutar el script COMO SI estuviéramos dentro de la carpeta.
        # sys.executable asegura que usamos el mismo entorno de Python actual (conda/venv).
        subprocess.run([sys.executable, cfg['script']], cwd=folder_path, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\nEl script falló con código de error: {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n🛑 Ejecución interrumpida por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {e}")

def main_menu():
    while True:
        #clear_console() # Descomentar si quieres limpiar pantalla en cada ciclo
        print("\n" + "="*40)
        print("   SELECTOR DE ARQUITECTURAS")
        print("="*40)
        print(f" [A] {CONFIG['A']['desc']}")
        print(f" [B] {CONFIG['B']['desc']}")
        print(" [Q] Salir")
        print("-" * 40)
        
        choice = input("Selecciona una opción (A/B/Q): ").strip().upper()

        if choice in ["A", "B"]:
            run_variant(choice)
            input("\nPresiona ENTER para volver al menú...")
        elif choice == "Q":
            print("Saliendo...")
            break
        else:
            print("Opción no válida.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nSaliendo...")