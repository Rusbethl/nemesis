import time
import os
import config
from processor import ejecutar_procesamiento_maestro # IMPORTA LA MISMA LÓGICA

def iniciar_vigilante():
    """
    Daemon de automatización. Monitorea la carpeta RAW y ejecuta el 
    procesamiento compartido para mantener el sistema autónomo.
    """
    print("="*60)
    print("👀 NÉMESIS WATCHER | EJECUCIÓN AUTOMATIZADA")
    print(f"📍 Vigilando entrada: {config.DIR_RAW}")
    print("="*60)

    try:
        while True:
            # Verifica si el Collector ha dejado nuevos archivos Parquet
            archivos = [f for f in os.listdir(config.DIR_RAW) if f.endswith(".parquet")]
            
            if archivos:
                print(f"\n🔔 Detectados {len(archivos)} lotes. Activando motor...")
                # Llama a la lógica idéntica que usa el modo manual
                ejecutar_procesamiento_maestro()
            
            # Tiempo de espera para optimizar recursos de CPU
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n🛑 Vigilante detenido manualmente.")

if __name__ == "__main__":
    iniciar_vigilante()