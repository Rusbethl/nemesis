import os
import hashlib
import config

def calcular_hash(path):
    """Calcula la firma digital MD5 del contenido del archivo."""
    hasher = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        return None

def filtro_seguridad_labeled():
    """
    Garantiza que la carpeta de entrenamiento no contenga duplicados internos.
    Es la última línea de defensa antes de alimentar a la red neuronal.
    """
    print("--- 🛡️ Ejecutando Filtro de Seguridad en Dataset de Entrenamiento ---")
    
    if not os.path.exists(config.DIR_LABELED):
        return

    archivos = [f for f in os.listdir(config.DIR_LABELED) if f.endswith(".parquet")]
    
    # Mapeo de contenido único para auditoría inmediata
    contenido_unico = {} # Estructura: { hash: nombre_archivo }
    eliminados = 0

    for arc in archivos:
        path_actual = os.path.join(config.DIR_LABELED, arc)
        h = calcular_hash(path_actual)
        
        if h is None: continue

        # Si el hash ya está registrado, el archivo actual es una copia redundante
        if h in contenido_unico:
            print(f"🗑️ Duplicado interno purgado: {arc}")
            os.remove(path_actual)
            eliminados += 1
        else:
            # Registro del nuevo archivo único
            contenido_unico[h] = arc

    print("-" * 50)
    print(f"✅ SEGURIDAD LABELED: {len(contenido_unico)} archivos únicos listos para la IA.")