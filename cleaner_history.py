import os
import hashlib
import polars as pl
import config

def calcular_hash(path):
    """Calcula la firma digital MD5 para identificar el contenido exacto."""
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def administrar_historico():
    """
    Escanea HISTORICO para eliminar duplicados y corregir etiquetas de archivos.
    Garantiza que el proyecto continúe de forma ligera y organizada.
    """
    print("\n" + "="*60)
    print("🧹 NÉMESIS VAULT CLEANER | SANEAMIENTO DE HISTÓRICO")
    print("="*60)

    if not os.path.exists(config.DIR_HISTORICO):
        print("❌ La carpeta HISTORICO no existe.")
        return

    archivos = [f for f in os.listdir(config.DIR_HISTORICO) if f.endswith(".parquet")]
    
    # Registro de firmas digitales para detectar duplicados
    firmas_unicas = {} # {hash: nombre_archivo}
    stats = {"duplicados": 0, "corruptos": 0, "renombrados": 0}

    for arc in archivos:
        path_actual = os.path.join(config.DIR_HISTORICO, arc)
        
        # 1. VERIFICACIÓN DE IDENTIDAD (Deduplicación)
        h = calcular_hash(path_actual)
        if not h:
            os.remove(path_actual)
            stats["corruptos"] += 1
            continue

        if h in firmas_unicas:
            # Si el contenido ya existe, eliminamos el archivo redundante
            os.remove(path_actual)
            stats["duplicados"] += 1
            continue
        
        firmas_unicas[h] = arc

        # 2. AUDITORÍA DE CONTENIDO (Clasificación Correcta)
        try:
            # Usamos scan_parquet para leer solo el encabezado (ahorro de memoria)
            schema = pl.scan_parquet(path_actual).collect_schema()
            columnas = schema.names()

            # Determinamos el prefijo real según las columnas presentes
            if "target" in columnas:
                prefijo_real = "labeled_"
            elif "OFI" in columnas or "CVD" in columnas:
                prefijo_real = "processed_"
            else:
                prefijo_real = "raw_"

            # Limpiamos el nombre base eliminando prefijos antiguos
            nombre_base = arc.replace("raw_", "").replace("processed_", "").replace("labeled_", "").replace("short_", "").replace("CORRUPT_", "")
            nuevo_nombre = f"{prefijo_real}{nombre_base}"
            path_nuevo = os.path.join(config.DIR_HISTORICO, nuevo_nombre)

            # 3. RENOMBRADO SI ES NECESARIO
            if path_actual != path_nuevo:
                if os.path.exists(path_nuevo):
                    os.remove(path_actual)
                    stats["duplicados"] += 1
                else:
                    os.rename(path_actual, path_nuevo)
                    stats["renombrados"] += 1

        except Exception:
            print(f"🗑️ Eliminando archivo ilegible: {arc}")
            os.remove(path_actual)
            stats["corruptos"] += 1

    # REPORTE FINAL
    print("-" * 60)
    print(f"✅ SANEAMIENTO COMPLETADO")
    print(f" 🔹 Archivos únicos preservados: {len(firmas_unicas)}")
    print(f" 🔹 Duplicados eliminados: {stats['duplicados']}")
    print(f" 🔹 Archivos corruptos purgados: {stats['corruptos']}")
    print(f" 🔹 Nombres corregidos/catalogados: {stats['renombrados']}")
    print("="*60)

if __name__ == "__main__":
    administrar_historico()