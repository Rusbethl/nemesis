import polars as pl
import os
import hashlib
import config

def calcular_hash(path):
    """
    Genera una huella digital MD5 del contenido del archivo.
    Esto permite identificar archivos idénticos aunque tengan nombres distintos.
    """
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            # Lectura por bloques para no saturar la memoria RAM con archivos grandes
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except:
        return None

def saneamiento_total():
    """
    Realiza una limpieza profunda en todo el ecosistema de carpetas.
    Elimina duplicados y archivos corruptos, y organiza el histórico.
    """
    print("--- 🧹 Iniciando Saneamiento de Datos Crítico ---")
    
    # 1. CONSTRUIR LISTA NEGRA: Identifica qué datos ya han sido procesados o entrenados
    hashes_listos = set()
    # TRAINED y LABELED se consideran las 'Fuentes de Verdad' finales
    for carpeta in [config.DIR_TRAINED, config.DIR_LABELED]:
        if not os.path.exists(carpeta): continue
        for arc in os.listdir(carpeta):
            path = os.path.join(carpeta, arc)
            if os.path.isfile(path):
                h = calcular_hash(path)
                if h: hashes_listos.add(h)
    
    print(f"🔍 Firmas únicas encontradas en destino: {len(hashes_listos)}")

    # 2. LIMPIAR ZONAS DE TRANSICIÓN (RAW, PROCESSED, HISTORICO)
    carpetas_a_limpiar = [config.DIR_RAW, config.DIR_PROCESSED, config.DIR_HISTORICO]
    
    eliminados_duplicados = 0
    eliminados_corruptos = 0
    re_clasificados = 0

    for carpeta in carpetas_a_limpiar:
        if not os.path.exists(carpeta): continue
        print(f"📁 Auditando integridad en: {os.path.basename(carpeta)}...")
        
        for arc in os.listdir(carpeta):
            path_actual = os.path.join(carpeta, arc)
            # Solo procesamos archivos Parquet válidos
            if not os.path.isfile(path_actual) or not arc.endswith(".parquet"): continue

            h = calcular_hash(path_actual)
            
            # A. PURGA DE DUPLICADOS: Si el contenido ya existe en destino, se borra el origen
            if h in hashes_listos:
                os.remove(path_actual)
                eliminados_duplicados += 1
                continue

            # B. VALIDACIÓN ESTRUCTURAL Y RE-CLASIFICACIÓN
            try:
                # Se escanea el esquema para verificar que el archivo sea legible
                schema = pl.scan_parquet(path_actual).collect_schema()
                columnas = schema.names()

                # C. LÓGICA DE ORGANIZACIÓN DEL HISTÓRICO
                # Asegura que cada archivo tenga el prefijo correcto según su grado de procesamiento
                if carpeta == config.DIR_HISTORICO:
                    if "target" in columnas:
                        pref = "labeled_"
                    elif "price" in columnas and len(columnas) > 5:
                        pref = "processed_"
                    else:
                        pref = "raw_"
                    
                    # Limpieza del nombre para evitar prefijos anidados (ej. raw_raw_archivo)
                    base = arc.replace("raw_", "").replace("processed_", "").replace("labeled_", "")
                    nuevo_nombre = f"{pref}{base}"
                    path_nuevo = os.path.join(config.DIR_HISTORICO, nuevo_nombre)
                    
                    if path_actual != path_nuevo:
                        if os.path.exists(path_nuevo):
                            os.remove(path_actual) # Elimina si el nombre destino ya existe
                            eliminados_duplicados += 1
                        else:
                            os.rename(path_actual, path_nuevo)
                            re_clasificados += 1
                
                # Actualizamos la lista de hashes para detectar duplicados dentro de la misma ejecución
                if h: hashes_listos.add(h)

            except Exception:
                # Manejo de archivos corruptos (Magic Bytes invalidos o truncados)
                print(f"🗑️ Eliminando archivo ilegible/corrupto: {arc}")
                os.remove(path_actual)
                eliminados_corruptos += 1

    print("-" * 50)
    print(f"✅ SANEAMIENTO FINALIZADO: El ecosistema está limpio y optimizado.")