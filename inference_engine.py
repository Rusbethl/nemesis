import polars as pl
import os
import torch
import torch.nn as nn
import time

class NemesisCNN(nn.Module):
    def __init__(self, input_dim):
        super(NemesisCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class InferenciaWatcher:
    def __init__(self, cerebro_path="MODELS/nemesis_v1.pth"):
        self.cerebro_path, self.modelo, self.ultima_act = cerebro_path, None, 0
        self.buffer, self.max_buffer = [], 500  
        self.meta = {"cvd": 0.0, "status": "Buscando Cerebro..."}
        self.detectar()

    def detectar(self):
        if not os.path.exists(self.cerebro_path): return
        try:
            mtime = os.path.getmtime(self.cerebro_path)
            if mtime > self.ultima_act:
                sd = torch.load(self.cerebro_path, map_location='cpu', weights_only=True)
                self.modelo = NemesisCNN(input_dim=sd['network.0.weight'].shape[1])
                self.modelo.load_state_dict(sd); self.modelo.eval()
                self.ultima_act, self.meta["status"] = mtime, "Cerebro Online"
        except: pass

    def recopilar(self, ev):
        self.detectar()
        self.buffer.append(ev)
        if len(self.buffer) > self.max_buffer: self.buffer.pop(0)
        if len(self.buffer) > 2:
            df = pl.DataFrame(self.buffer)
            df = df.with_columns([
                pl.col("quantity").cast(pl.Float64), 
                pl.when(pl.col("side")=="buy").then(pl.col("quantity")).otherwise(-pl.col("quantity")).alias("v")
            ])
            self.meta["cvd"] = df["v"].sum()

if __name__ == "__main__":
    watcher = InferenciaWatcher()
    while True:
        # Ejemplo de prueba
        watcher.recopilar({"price": 90000, "quantity": 0.01, "side": "buy"})
        print(f"CVD: {watcher.meta['cvd']:.4f} | Status: {watcher.meta['status']}")
        time.sleep(1)