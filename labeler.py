import polars as pl
import os
import shutil
import config
import time

def etiquetar_datos(look_ahead=50):
    """
    Transforma archivos de PROCESSED en datasets de entrenamiento en LABELED.
    Aplica la etiqueta de éxito basada en un horizonte de 50 eventos de mercado.
    """
    # 1. Identificación de archivos procesados pendientes de etiqueta
    archivos = [f for f in os.listdir(config.DIR_PROCESSED) if f.endswith(".parquet")]
    
    if not archivos:
        return 0

    count_nuevos = 0

    for arc in archivos:
        path_in = os.path.join(config.DIR_PROCESSED, arc)
        path_out = os.path.join(config.DIR_LABELED, f"ready_{arc.replace('proc_', '')}")
        
        # Nombre estandarizado para la trazabilidad en HISTORICO
        nombre_hist = f"processed_{arc.replace('proc_', '').replace('processed_', '')}"
        path_hist = os.path.join(config.DIR_HISTORICO, nombre_hist)

        try:
            # Carga eficiente del DataFrame procesado (contiene OFI/CVD)
            df = pl.read_parquet(path_in)
            
            # --- VALIDACIÓN DE DIMENSIÓN ---
            # Se requiere que el lote sea mayor al horizonte para poder 'ver el futuro'
            if len(df) <= look_ahead:
                print(f"⚠️ Lote insuficiente para look_ahead=50, omitiendo: {arc}")
                if not os.path.exists(path_hist):
                    shutil.move(path_in, os.path.join(config.DIR_HISTORICO, f"short_{nombre_hist}"))
                else:
                    os.remove(path_in)
                continue

            # --- GENERACIÓN DE ETIQUETA Y (Target) ---
            # Desplazamos el precio 50 posiciones al pasado para comparar 
            # el precio actual contra el que habrá en 50 eventos.
            df = df.with_columns([
                pl.col("price").shift(-look_ahead).alias("future_price")
            ])

            # Etiquetado binario: 1 (Sube) o 0 (Baja/Igual)
            df = df.with_columns([
                (pl.col("future_price") > pl.col("price")).cast(pl.Float32).alias("target")
            ])

            # LIMPIEZA CRÍTICA: Eliminamos las últimas 50 filas (donde no hay futuro conocido)
            # Esto evita que la IA aprenda de datos nulos (NaN)
            df = df.filter(pl.col("target").is_not_null())

            if len(df) > 0:
                # Almacenamiento persistente en LABELED (Listo para la CNN-1D)
                df.write_parquet(path_out, compression="snappy")
                count_nuevos += 1
                
                # --- SISTEMA DE ARCHIVADO Y AHORRO DE DATOS ---
                # El archivo original se mueve a HISTORICO para liberar espacio en PROCESSED
                # y servir como respaldo inalterable de la investigación.
                if os.path.exists(path_hist):
                    os.remove(path_in) 
                else:
                    shutil.move(path_in, path_hist)
                
                print(f"✅ Lote etiquetado con éxito: {nombre_hist}")

        except Exception as e:
            print(f"❌ Fallo en el etiquetado del lote {arc}: {e}")
            
    return count_nuevos

if __name__ == "__main__":
    print("="*60)
    print("🤖 NÉMESIS LABELER | MODO: ALTA FRECUENCIA (50 EVENTOS)")
    print("="*60)

    try:
        while True:
            # El bucle automático permite la recolección y etiquetado 24/7
            nuevos = etiquetar_datos(look_ahead=50)
            if nuevos > 0:
                print(f"📊 {nuevos} archivos nuevos disponibles para entrenamiento.")
            
            # Pausa de 30 segundos para optimizar el consumo de recursos
            time.sleep(30) 
            
    except KeyboardInterrupt:
        print("\n🛑 Servicio de etiquetado detenido manualmente.")