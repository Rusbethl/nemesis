import os

# Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpetas del Sistema (Respetando tu estructura)
DIR_RAW = os.path.join(BASE_DIR, "RAW")
DIR_PROCESSED = os.path.join(BASE_DIR, "PROCESSED")
DIR_HISTORICO = os.path.join(BASE_DIR, "HISTORICO")
DIR_LABELED = os.path.join(BASE_DIR, "LABELED")
DIR_TRAINED = os.path.join(BASE_DIR, "TRAINED")
DIR_MODELS = os.path.join(BASE_DIR, "MODELS")



# Asegurar carpetas para no perder datos [cite: 2025-12-23]
folders = [DIR_RAW, DIR_PROCESSED, DIR_LABELED, DIR_HISTORICO, DIR_MODELS, DIR_TRAINED]
for folder in folders:
    os.makedirs(folder, exist_ok=True)