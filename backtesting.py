import polars as pl
import polars.selectors as cs
import os
import torch
import config
from model import NemesisCNN
import numpy as np

def ejecutar_backtesting_pro(threshold=0.85):
    path_modelo = os.path.join(config.DIR_MODELS, "nemesis_v1.pth")
    if not os.path.exists(path_modelo):
        print("❌ No se encontró el modelo.")
        return

    archivos = [f for f in os.listdir(config.DIR_TRAINED) if f.endswith(".parquet")]
    if not archivos:
        print("📭 Carpeta TRAINED vacía.")
        return

    modelo = None
    balance, balance_inicial = 100.0, 100.0
    stats = {"wins": 0, "losses": 0, "max_balance": 100.0, "max_drawdown": 0.0}
    history = []

    print(f"🔬 Iniciando Auditoría PRO (Confianza > {threshold})...")

    for arc in archivos[:150]: # Analizamos 150 lotes para mayor detalle
        try:
            df = pl.read_parquet(os.path.join(config.DIR_TRAINED, arc))
            X_df = df.drop(["target", "price", "future_price"]).select(cs.numeric())
            y_real = df["target"].to_numpy()

            if modelo is None:
                input_dim = len(X_df.columns)
                modelo = NemesisCNN(input_dim=input_dim)
                modelo.load_state_dict(torch.load(path_modelo, weights_only=True))
                modelo.eval()
                print(f"✅ Cerebro cargado con {input_dim} indicadores.")

            with torch.no_grad():
                X_tensor = torch.tensor(X_df.to_pandas().values.astype('float32'))
                preds = modelo(X_tensor).numpy().flatten()

            # Lógica de señales de alta convicción
            mask = (preds > threshold) | (preds < (1 - threshold))
            
            if mask.any():
                pred_bin = (preds[mask] > 0.5).astype(int)
                real_bin = y_real[mask]

                for i in range(len(pred_bin)):
                    if pred_bin[i] == real_bin[i]:
                        balance *= 1.01  # +1% profit
                        stats["wins"] += 1
                    else:
                        balance *= 0.99  # -1% loss (Stop Loss)
                        stats["losses"] += 1
                    
                    # Cálculo de Drawdown
                    stats["max_balance"] = max(stats["max_balance"], balance)
                    dd = (stats["max_balance"] - balance) / stats["max_balance"] * 100
                    stats["max_drawdown"] = max(stats["max_drawdown"], dd)
                    history.append(balance)

        except: continue

    # --- REPORTE DETALLADO ---
    total_ops = stats["wins"] + stats["losses"]
    print("\n" + "═"*60)
    print("🛡️  NÉMESIS PERFORMANCE REPORT")
    print("═"*60)
    if total_ops > 0:
        win_rate = (stats["wins"] / total_ops) * 100
        roi = ((balance / balance_inicial) - 1) * 100
        
        print(f"📊 MÉTRICAS DE PRECISIÓN:")
        print(f"   🔹 Win Rate Elite:    {win_rate:.2f}%")
        print(f"   🔹 Total Operaciones: {total_ops}")
        print(f"   🔹 Aciertos/Fallos:   {stats['wins']} / {stats['losses']}")
        
        print(f"\n💰 MÉTRICAS FINANCIERAS:")
        print(f"   🔹 Retorno (ROI):     {roi:.2f}%")
        print(f"   🔹 Balance Final:     {balance:.2f} USD")
        print(f"   🔹 Max Drawdown:      -{stats['max_drawdown']:.2f}%")
        
        status = "🟢 RENTABLE" if balance > balance_inicial else "🔴 NO RENTABLE"
        print(f"\nESTADO DEL SISTEMA: {status}")
    else:
        print("⚠️ No se generaron señales suficientes con este umbral.")
    print("═"*60)

if __name__ == "__main__":
    ejecutar_backtesting_pro()
    input("\nPresiona ENTER para cerrar...")