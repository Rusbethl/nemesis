import polars as pl
import polars.selectors as cs
import os
import shutil
import torch
import config
from model import NemesisCNN

def entrenar_incremental():
    """
    Motor de Aprendizaje Aumentado.
    Carga la experiencia previa (nemesis_v1.pth) y le suma los datos nuevos.
    """
    path_modelo = os.path.join(config.DIR_MODELS, "nemesis_v1.pth")
    if not os.path.exists(config.DIR_MODELS): os.makedirs(config.DIR_MODELS)
    
    # Identificación de nuevos lotes etiquetados
    archivos = [f for f in os.listdir(config.DIR_LABELED) if f.endswith(".parquet")]
    if not archivos:
        print("📭 No hay datos nuevos. El cerebro se mantiene en espera.")
        return

    modelo = None

    for arc in archivos:
        path_in = os.path.join(config.DIR_LABELED, arc)
        path_out = os.path.join(config.DIR_TRAINED, arc)

        try:
            # Lectura del lote de entrenamiento
            df = pl.read_parquet(path_in)
            
            # --- SELECCIÓN DE CARACTERÍSTICAS ---
            # Filtramos solo columnas numéricas útiles (OFI, CVD, etc.)
            X_df = df.drop(["target", "price", "future_price"])
            X_df = X_df.select(cs.numeric()) 
            y_df = df.select(pl.col("target")).cast(pl.Float32)

            # --- LÓGICA DE PERSISTENCIA (OBJETIVO: NO EMPEZAR DE CERO) ---
            if modelo is None:
                input_dim = len(X_df.columns)
                modelo = NemesisCNN(input_dim=input_dim)
                # Si ya existe un archivo de modelo, cargamos el conocimiento previo
                if os.path.exists(path_modelo):
                    modelo.load_state_dict(torch.load(path_modelo, weights_only=True))
                    print(f"✅ Conocimiento previo cargado. Continuando aprendizaje...")
                else:
                    print(f"🆕 Iniciando aprendizaje desde el primer lote.")

            # Entrenamiento incremental (Ajuste de pesos con el lote actual)
            loss = modelo.fit(X_df.to_pandas(), y_df.to_pandas())
            
            # --- MOVIMIENTO Y AHORRO DE DATOS ---
            # El archivo se mueve a TRAINED para indicar que ya fue 'aprendido' por la IA
            if os.path.exists(path_out): os.remove(path_in)
            else: shutil.move(path_in, path_out)
            
            print(f"✔️ Lote {arc} integrado exitosamente | Pérdida: {loss:.6f}")

        except Exception as e:
            print(f"❌ Error crítico en el aprendizaje del lote {arc}: {e}")

    # GUARDADO FINAL DEL ESTADO
    # Asegura que todo lo aprendido se guarde para la próxima vez que abras el proyecto.
    if modelo is not None:
        torch.save(modelo.state_dict(), path_modelo)
        print(f"💾 Cerebro de Némesis actualizado y guardado en: {path_modelo}")

if __name__ == "__main__":
    print("🤖 NÉMESIS TRAINER | SISTEMA DE APRENDIZAJE AUMENTADO")
    entrenar_incremental()