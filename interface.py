import dash
from dash import dcc, html, Input, Output
import os
import torch
import torch.nn as nn
import json
import sys
import time
import asyncio
from binance import ThreadedWebsocketManager

# PARCHE PARA WINDOWS
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============================================================
# ARQUITECTURA DEL MODELO
# Idéntica a model.py/train.py. El checkpoint nemesis_v1.pth se
# inspeccionó directamente: network.0.weight tiene shape (64, 3),
# es decir input_dim=3. No se puede cambiar sin reentrenar.
# ============================================================
class NemesisCNN(nn.Module):
    def __init__(self, input_dim):
        super(NemesisCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)


MODEL_PATH = "MODELS/nemesis_v1.pth"
MEMORY_PATH = "MODELS/nemesis_memory.json"
SYMBOL = "BTCUSDT"

# ------------------------------------------------------------------
# QUÉ FEATURES USA REALMENTE EL MODELO (verificado contra processor.py
# y train.py, no contra la documentación de la tesis, porque el código
# y la tesis no coinciden en este punto):
#
#   columna "OFI" en el dataset de entrenamiento = delta firmado de
#   CADA TRADE individual (+quantity si compra a mercado, -quantity si
#   venta a mercado). NO es el Order Flow Imbalance de Libro de Órdenes
#   (Bid/Ask) que describe el marco teórico — esa versión nunca se
#   calculó en ningún archivo del pipeline; bids/asks se capturan en
#   collector.py pero no se usan en ningún lado.
#
#   columna "CVD" = suma acumulada de esa misma columna OFI, DENTRO DE
#   CADA LOTE que guarda collector.py (~1000 eventos). Se reinicia en
#   cada lote, no es un acumulado de toda la sesión.
#
# Esta interfaz solo muestra y usa estas 3 columnas, en este orden y
# en esta escala, porque son las únicas 3 que el modelo (input_dim=3)
# recibió durante el entrenamiento. No se inventan ni muestran métricas
# adicionales (alertas de "absorción", "ballenas", etc.) que no salgan
# de un cálculo que el software realmente ejecuta.
# ------------------------------------------------------------------
CVD_RESET_WINDOW = 1000      # igual al tamaño de lote de collector.py
CONFIDENCE_THRESHOLD = 0.85  # igual que backtesting.py y la tesis (3.5)


class NemesisEngine:
    def __init__(self):
        self.modelo, self.last_mtime = None, 0
        self.model_input_dim = None
        self.trades_count = 0
        self.last_update_time = time.time()
        self.price, self.prev_price = 0.0, 0.0

        # --- Las 3 señales EXACTAS que entran al modelo ---
        self.last_quantity = 0.0
        self.last_ofi = 0.0
        self.cvd_window = 0.0
        self.events_in_window = 0

        self.prob = None
        self.signal = "SIN DATOS"
        self.instruction = "STANDBY"
        self.status_msg = "SISTEMA INICIANDO..."
        self.running, self.twm = True, None
        self.reconnecting = False

        self.cargar_memoria_json()
        self.start_socket()
        self.check_model()

    def guardar_memoria_json(self):
        try:
            data = {"trades_count": self.trades_count, "timestamp": time.time()}
            with open(MEMORY_PATH, "w") as f: json.dump(data, f)
        except: pass

    def cargar_memoria_json(self):
        if os.path.exists(MEMORY_PATH):
            try:
                with open(MEMORY_PATH, "r") as f:
                    data = json.load(f)
                    self.trades_count = data.get("trades_count", 0)
            except: pass

    def start_socket(self):
        if self.reconnecting: return
        self.reconnecting = True
        self.status_msg = "REINICIANDO MOTOR..."
        try:
            if self.twm:
                self.twm.stop()
                time.sleep(2)
                self.twm = None
        except: pass

        try:
            self.twm = ThreadedWebsocketManager()
            self.twm.start()
            self.twm.start_aggtrade_socket(callback=self.handle_socket, symbol=SYMBOL.upper())
            self.status_msg = "CONECTADO A BINANCE"
            self.last_update_time = time.time()
        except:
            self.status_msg = "FALLO DE RED. REINTENTANDO..."
        finally:
            self.reconnecting = False

    def check_reconnection(self):
        if self.running and not self.reconnecting:
            if (time.time() - self.last_update_time > 10):
                self.start_socket()

    def handle_socket(self, msg):
        try:
            if msg.get('e') == 'error':
                self.last_update_time = 0
                return

            p, q = float(msg['p']), float(msg['q'])
            side = "sell" if msg['m'] else "buy"
            self.last_update_time = time.time()

            if self.price != p: self.prev_price, self.price = self.price, p

            # --- Réplica exacta de la ingeniería de features de processor.py ---
            ofi_evento = q if side == "buy" else -q
            self.last_quantity = q
            self.last_ofi = ofi_evento
            self.cvd_window += ofi_evento
            self.events_in_window += 1
            if self.events_in_window >= CVD_RESET_WINDOW:
                self.cvd_window = 0.0
                self.events_in_window = 0

            self.trades_count += 1
            if self.trades_count % 100 == 0: self.guardar_memoria_json()
            self.inferir()
        except: self.last_update_time = 0

    def check_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime > self.last_mtime:
                    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
                    self.model_input_dim = sd['network.0.weight'].shape[1]
                    self.modelo = NemesisCNN(input_dim=self.model_input_dim)
                    self.modelo.load_state_dict(sd); self.modelo.eval()
                    self.last_mtime = mtime
            except: pass

    def inferir(self):
        self.check_model()

        if self.modelo is None:
            self.signal, self.prob, self.instruction = "SIN CEREBRO", None, "STANDBY"
            return

        try:
            # --- Vector EXACTO [quantity, OFI, CVD], mismo orden/escala que train.py ---
            feats = [self.last_quantity, self.last_ofi, self.cvd_window]
            input_t = torch.tensor([feats], dtype=torch.float32)
            with torch.no_grad():
                prob = self.modelo(input_t).item()
            self.prob = prob

            # --- Mismo umbral de confianza (0.85) que backtesting.py ---
            if prob >= CONFIDENCE_THRESHOLD:
                self.signal = "LONG 🔼"
                self.instruction = "MANTENER"
            elif prob <= (1 - CONFIDENCE_THRESHOLD):
                self.signal = "SHORT 🔽"
                self.instruction = "MANTENER"
            else:
                self.signal = "SIN SEÑAL"
                self.instruction = "NO OPERAR / ESPERAR"
        except Exception:
            self.signal, self.prob, self.instruction = "ERROR IA", None, "STANDBY"


engine = NemesisEngine()
app = dash.Dash(__name__)

DARK_CARD = {'background': '#15191d', 'padding': '20px', 'borderRadius': '10px', 'border': '1px solid #2d3436', 'textAlign': 'center', 'color': 'white'}
NOTE_STYLE = {'background': '#1c2227', 'padding': '20px', 'borderRadius': '10px', 'border': '1px solid #2d3436', 'color': '#aab', 'marginTop': '20px', 'fontSize': '13px', 'lineHeight': '1.6'}

app.layout = html.Div(style={'backgroundColor': '#0b0e11', 'minHeight': '100vh', 'padding': '40px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[

    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'borderBottom': '1px solid #333', 'paddingBottom': '20px'}, children=[
        html.H1("NEMESIS T-CORE", style={'color': '#00ffcc', 'margin': '0', 'letterSpacing': '3px'}),
        html.Button("🛑 DETENER SERVICIO", id="btn-stop", n_clicks=0, style={'background': '#ff4757', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'fontWeight': 'bold', 'cursor': 'pointer'})
    ]),

    # BLOQUE 1: lo que produce el modelo
    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px', 'marginTop': '30px'}, children=[
        html.Div(style=DARK_CARD, children=[html.P("PRECIO BTC/USDT", style={'color': '#888'}), html.H2(id="out-precio", style={'color': '#00d1ff'})]),
        html.Div(style=DARK_CARD, children=[html.P("PROBABILIDAD DEL MODELO", style={'color': '#888'}), html.H2(id="out-prob")]),
        html.Div(style=DARK_CARD, children=[html.P(f"SEÑAL (umbral {int(CONFIDENCE_THRESHOLD*100)}%)", style={'color': '#888'}), html.H2(id="out-signal")]),
        html.Div(style=DARK_CARD, children=[html.P("INSTRUCCIÓN", style={'color': '#888'}), html.H2(id="out-instruction", style={'color': '#ffa502'})]),
    ]),

    # BLOQUE 2: las 3 features crudas que entran al modelo, sin transformar
    html.Div(style={'marginTop': '10px'}, children=[
        html.P("ENTRADA AL MODELO (exactamente lo que ve la red, sin transformar)", style={'color': '#666', 'fontSize': '12px', 'letterSpacing': '1px', 'marginTop': '20px'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px'}, children=[
            html.Div(style=DARK_CARD, children=[html.P("QUANTITY (último trade, BTC)", style={'color': '#888'}), html.H3(id="out-qty")]),
            html.Div(style=DARK_CARD, children=[html.P("OFI (delta de trade firmado)", style={'color': '#888'}), html.H3(id="out-ofi")]),
            html.Div(style=DARK_CARD, children=[html.P(f"CVD (ventana de {CVD_RESET_WINDOW} eventos)", style={'color': '#888'}), html.H3(id="out-cvd")]),
        ]),
    ]),

    # NOTA TÉCNICA — solo hechos verificables contra el código, sin narrativa
    html.Div(style=NOTE_STYLE, children=[
        html.P([
            html.B("Nota: "), "\"OFI\" y \"CVD\" aquí son los nombres literales de las columnas con las que se entrenó ",
            html.Code("nemesis_v1.pth"), " (ver ", html.Code("processor.py"), "), no el Order Flow Imbalance de Libro de ",
            "Órdenes descrito en el marco teórico de la tesis — esa versión requiere parsear bids/asks, lo cual el pipeline ",
            "actual no hace. Esta interfaz no muestra ningún indicador que el software no calcule realmente."
        ])
    ]),

    html.Div(id="out-status", style={'textAlign': 'center', 'marginTop': '30px', 'color': '#00ffcc', 'fontSize': '12px', 'opacity': '0.7'}),
    dcc.Interval(id='refresh', interval=500, n_intervals=0)
])


@app.callback(
    [Output("out-precio", "children"), Output("out-prob", "children"), Output("out-prob", "style"),
     Output("out-signal", "children"), Output("out-signal", "style"), Output("out-instruction", "children"),
     Output("out-qty", "children"), Output("out-ofi", "children"), Output("out-ofi", "style"),
     Output("out-cvd", "children"), Output("out-cvd", "style"), Output("out-status", "children")],
    [Input("refresh", "n_intervals"), Input("btn-stop", "n_clicks")]
)
def update_dashboard(n, n_clicks):
    if n_clicks > 0:
        engine.guardar_memoria_json()
        os._exit(0)
    engine.check_reconnection()

    c_signal = '#2ecc71' if "LONG" in engine.signal else '#ff4757' if "SHORT" in engine.signal else '#fff'
    c_ofi = '#2ecc71' if engine.last_ofi > 0 else '#ff4757' if engine.last_ofi < 0 else '#fff'
    c_cvd = '#2ecc71' if engine.cvd_window > 0 else '#ff4757' if engine.cvd_window < 0 else '#fff'
    prob_txt = f"{engine.prob*100:.1f}%" if engine.prob is not None else "—"

    return (
        f"${engine.price:,.2f}" if engine.price > 0 else "CONECTANDO...",
        prob_txt, {'color': '#00d1ff', 'fontSize': '32px', 'fontWeight': 'bold'},
        engine.signal, {'color': c_signal, 'fontSize': '32px', 'fontWeight': 'bold'},
        engine.instruction,
        f"{engine.last_quantity:.4f}",
        f"{engine.last_ofi:+.4f}", {'color': c_ofi},
        f"{engine.cvd_window:+.4f}", {'color': c_cvd},
        f"STATUS: {engine.status_msg} | TRADES: {engine.trades_count}"
    )


if __name__ == "__main__":
    app.run(debug=False, port=8050, use_reloader=False)
