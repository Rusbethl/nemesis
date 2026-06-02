import os
import json
import polars as pl
import config
from datetime import datetime

# Archivos de persistencia para la integridad del sistema
STATE_FILE = "nemesis_metadata.json"  # Generado por el Collector
AUDIT_LOG = "last_audit_report.json"   # Memoria del auditor para evitar re-procesos

def auditoria():
    """
    Realiza un diagnóstico profundo del estado de los datos en el sistema.
    Verifica la integridad de los archivos Parquet, detecta archivos corruptos
    y sincroniza el conteo físico con los metadatos globales del proyecto.
    """
    print("\n" + "="*60)
    print(f"🛡️ NÉMESIS SYSTEM | AUDITORÍA DE INTEGRIDAD Y PROGRESO")
    print(f"📅 Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # --- Sincronización con Metadatos del Collector ---
    # Este paso es vital para cumplir con el objetivo de continuidad del proyecto.
    collector_total = 0
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                collector_total = json.load(f).get("sesion_total", 0)
        except:
            print("⚠️ Advertencia: No se pudo leer el archivo de metadatos del Collector.")

    # --- Interacción de Usuario para Optimización de Recursos ---
    # En proyectos de 10M de eventos, escanear el HISTORICO en cada ejecución es ineficiente.
    confirmar_hist = input("¿Deseas realizar el conteo profundo de eventos en HISTORICO? (s/n): ").lower()
    procesar_hist = (confirmar_hist == 's')
    print("-" * 60)

    # Definición de la estructura de directorios del ecosistema Némesis
    carpetas = {
        "RAW (Entrada)": config.DIR_RAW,
        "PROCESSED (Limpio)": config.DIR_PROCESSED,
        "LABELED (Por Entrenar)": config.DIR_LABELED,
        "TRAINED (IA Completada)": config.DIR_TRAINED,
        "HISTORICO (Respaldo)": config.DIR_HISTORICO
    }

    meta_final = 10000000  # Objetivo de la Tesis: 10 millones de eventos
    eventos_calidad = 0    # Solo se cuentan datos útiles para el entrenamiento (LABELED + TRAINED)
    total_archivos = 0
    metricas_audit = {}    # Diccionario para guardar el reporte actual

    for nombre, ruta in carpetas.items():
        if not os.path.exists(ruta): 
            print(f"📁 {nombre:23} | ❌ Carpeta no encontrada")
            continue
        
        # Filtramos solo archivos .parquet, ignorando archivos temporales .tmp
        archivos = [f for f in os.listdir(ruta) if f.endswith(".parquet")]
        eventos_carpeta = 0
        corruptos = 0
        
        es_historico = "HISTORICO" in nombre
        
        # Lógica de escaneo eficiente usando Lazy Execution de Polars
        if es_historico and not procesar_hist:
            eventos_display = "Saltado (Optimización de tiempo)"
        else:
            for f in archivos:
                try:
                    # pl.scan_parquet no carga el archivo en RAM, solo lee los metadatos
                    # Esto permite auditar millones de eventos en segundos.
                    count = pl.scan_parquet(os.path.join(ruta, f)).select(pl.len()).collect().item()
                    eventos_carpeta += count
                except Exception:
                    corruptos += 1
                    continue
            eventos_display = f"{eventos_carpeta:,} eventos"

        total_archivos += len(archivos)
        
        # Clasificación de eventos para el resumen de entrenamiento
        if "LABELED" in nombre or "TRAINED" in nombre:
            eventos_calidad += eventos_carpeta
        
        # Almacenamos métricas para la persistencia del reporte
        metricas_audit[nombre] = {"archivos": len(archivos), "eventos": eventos_carpeta}

        # Impresión de resultados por módulo
        status_corruptos = f" | 💥 {corruptos} corruptos" if corruptos > 0 else ""
        print(f"📁 {nombre:23} | {len(archivos):4} archivos | {eventos_display}{status_corruptos}")

    # --- Cálculo de Progreso y Salud del Sistema ---
    progreso = (eventos_calidad / meta_final) * 100
    
    print("-" * 60)
    print(f"📊 RESUMEN DE PROGRESO (Meta 10M):")
    print(f"   🔹 Archivos totales en disco: {total_archivos}")
    print(f"   🔹 Eventos ÚNICOS de calidad: {eventos_calidad:,} / {meta_final:,} ({progreso:.2f}%)")
    
    # --- Validación Cruzada con el Collector ---
    # Comprobamos si el registro del sistema coincide con la realidad de los archivos RAW.
    if collector_total > 0:
        desfase = collector_total - metricas_audit.get("RAW (Entrada)", {}).get("eventos", 0)
        if desfase != 0:
            print(f"   ⚠️ Desfase detectado: {desfase:,} eventos están en metadatos pero no en RAW (¿Archivos movidos?)")

    # --- Persistencia del Reporte de Auditoría ---
    # Guardamos los resultados para que otros módulos (como el Trainer) puedan consultarlos.
    with open(AUDIT_LOG, 'w') as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "progreso_ia": progreso,
            "total_eventos_calidad": eventos_calidad,
            "detalles": metricas_audit
        }, f, indent=4)

    if eventos_calidad >= meta_final:
        print("\n🎯 ¡META ALCANZADA! El cerebro de Némesis tiene datos suficientes para el entrenamiento final.")
    else:
        faltantes = meta_final - eventos_calidad
        print(f"   💡 Faltan {faltantes:,} eventos para el objetivo de 10M.")
    print("="*60)

if __name__ == "__main__":
    auditoria()
    input("\nPresiona ENTER para finalizar y guardar reporte...")