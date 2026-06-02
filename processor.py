import polars as pl
import os
import shutil
import config

def ejecutar_procesamiento_maestro():
    """
    Función Núcleo: Realiza la lectura, cálculo de indicadores y el 
    movimiento físico de los archivos para garantizar la limpieza de RAW.
    """
    # 1. Identificación de archivos en la zona de entrada (RAW)
    archivos = [f for f in os.listdir(config.DIR_RAW) if f.endswith(".parquet")]
    if not archivos: 
        return

    for arc in archivos:
        path_in = os.path.join(config.DIR_RAW, arc)
        path_out = os.path.join(config.DIR_PROCESSED, f"proc_{arc}")
        
        # Estandarización del nombre para el respaldo histórico de la investigación
        nombre_hist = f"raw_{arc.replace('raw_', '')}"
        path_hist = os.path.join(config.DIR_HISTORICO, nombre_hist)
        
        try:
            # Lectura optimizada con Polars (Motor de alta velocidad para 10M de eventos)
            df = pl.read_parquet(path_in)
            
            # --- VALIDACIÓN DE INTEGRIDAD ---
            if "side" not in df.columns:
                print(f"⚠️ Archivo sin datos de Side, archivando sin procesar: {arc}")
                if not os.path.exists(path_hist): shutil.move(path_in, path_hist)
                else: os.remove(path_in)
                continue

            # --- INGENIERÍA DE CARACTERÍSTICAS (OFI & CVD) ---
            # Transformación de datos para análisis de presión de mercado
            df = df.with_columns([
                pl.col("price").cast(pl.Float64, strict=False),
                pl.col("quantity").cast(pl.Float64, strict=False)
            ])

            # Cálculo de Order Flow Imbalance (OFI)
            df = df.with_columns(
                pl.when(pl.col("side") == "buy").then(pl.col("quantity"))
                .when(pl.col("side") == "sell").then(-pl.col("quantity"))
                .otherwise(0).alias("OFI")
            )

            # Cálculo de Cumulative Volume Delta (CVD)
            df = df.with_columns(pl.col("OFI").cum_sum().alias("CVD"))

            # --- PERSISTENCIA Y CIERRE ---
            # Guardado del dato procesado con compresión Snappy (Ahorro de disco)
            df.write_parquet(path_out, compression="snappy")
            
            # MOVIMIENTO A HISTORICO: Se asegura que RAW quede vacío para la siguiente carga
            if os.path.exists(path_hist):
                os.remove(path_in) 
            else:
                shutil.move(path_in, path_hist)
                
            print(f"✅ Procesado y Archivado en HISTORICO: {nombre_hist}")

        except Exception as e:
            # Gestión de archivos corruptos por cortes de energía o red
            if "magic bytes" in str(e).lower():
                print(f"🗑️ Detectado archivo corrupto, aislando en histórico.")
                shutil.move(path_in, os.path.join(config.DIR_HISTORICO, f"CORRUPT_{arc}"))

if __name__ == "__main__":
    # Punto de entrada para ejecución MANUAL por el investigador
    ejecutar_procesamiento_maestro()