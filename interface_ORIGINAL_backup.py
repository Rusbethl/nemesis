import dash
from dash import dcc, html, Input, Output
import polars as pl
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

class NemesisEngine:
    def __init__(self):
        self.buffer, self.max_buffer = [], 500 
        self.modelo, self.last_mtime = None, 0
        self.trades_count, self.last_update_time = 0, time.time()
        self.price, self.prev_price = 0.0, 0.0
        self.session_delta, self.session_total_vol = 0.0, 0.0
        self.current_cvd, self.cvd_pct, self.ofi_pct = 0.0, 0.0, 0.0
        self.prediction, self.close_instruction = "INICIANDO...", "STANDBY"
        self.alert_manipulation, self.alert_whales = "ESTABLE", "NORMAL"
        self.status_msg = "SISTEMA INICIANDO..."
        self.running, self.twm = True, None
        self.reconnecting = False
        
        self.cargar_memoria_json()
        self.start_socket()
        self.check_model()

    def guardar_memoria_json(self):
        try:
            data = {"session_delta": self.session_delta, "session_total_vol": self.session_total_vol, 
                    "trades_count": self.trades_count, "timestamp": time.time()}
            with open(MEMORY_PATH, "w") as f: json.dump(data, f)
        except: pass

    def cargar_memoria_json(self):
        if os.path.exists(MEMORY_PATH):
            try:
                with open(MEMORY_PATH, "r") as f:
                    data = json.load(f)
                    self.session_delta = data.get("session_delta", 0.0)
                    self.session_total_vol = data.get("session_total_vol", 0.0)
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
            
            t_delta = q if side == "buy" else -q
            self.session_delta += t_delta
            self.session_total_vol += q
            self.buffer.append({"q": q, "side": side})
            if len(self.buffer) > self.max_buffer: self.buffer.pop(0)
            
            self.trades_count += 1
            if self.trades_count % 100 == 0: self.guardar_memoria_json()
            self.logica_nemesis()
        except: self.last_update_time = 0

    def check_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                mtime = os.path.getmtime(MODEL_PATH)
                if mtime > self.last_mtime:
                    sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
                    self.modelo = NemesisCNN(input_dim=sd['network.0.weight'].shape[1])
                    self.modelo.load_state_dict(sd); self.modelo.eval()
                    self.last_mtime = mtime
            except: pass

    def logica_nemesis(self):
        if len(self.buffer) < 10: return
        self.check_model()
        df = pl.DataFrame(self.buffer)
        df = df.with_columns([pl.when(pl.col("side")=="buy").then(pl.col("q")).otherwise(-pl.col("q")).alias("v")])
        self.ofi_pct = (df["v"].sum() / df["q"].sum()) * 100 if df["q"].sum() > 0 else 0
        self.cvd_pct = (self.session_delta / self.session_total_vol) * 100 if self.session_total_vol > 0 else 0
        
        p_diff = abs(self.price - self.prev_price)
        self.alert_manipulation = "⚠️ ABSORCIÓN CRÍTICA" if abs(self.ofi_pct) > 25 and p_diff < (self.price * 0.00005) else "ESTABLE"
        
        if self.ofi_pct > 20 and self.price < self.prev_price:
            self.alert_whales = "🐋 COMPRA AGRESIVA" 
        elif self.ofi_pct < -20 and self.price > self.prev_price:
            if self.cvd_pct > -5:
                self.alert_whales = "🐋 VENTA AGRESIVA"
            else:
                self.alert_whales = "NORMAL (BAJISTA FUERTE)"
        else:
            self.alert_whales = "NORMAL"

        if self.modelo:
            try:
                exp = self.modelo.network[0].in_features
                feats = [self.ofi_pct, self.cvd_pct]
                if len(feats) < exp: feats += [0.0] * (exp - len(feats))
                input_t = torch.tensor([feats[:exp]], dtype=torch.float32)
                with torch.no_grad():
                    prob = self.modelo(input_t).item()
                self.prediction = "LONG 🔼" if prob > 0.65 else "SHORT 🔽" if prob < 0.35 else "NEUTRAL"
                self.close_instruction = "MANTENER" if prob > 0.65 or prob < 0.35 else "CERRAR POSICIÓN"
            except: self.prediction = "ERROR IA"
        else: self.prediction = "SIN CEREBRO"

engine = NemesisEngine()
app = dash.Dash(__name__)

# ESTILOS DE UI
DARK_CARD = {'background': '#15191d', 'padding': '20px', 'borderRadius': '10px', 'border': '1px solid #2d3436', 'textAlign': 'center', 'color': 'white'}
MANUAL_STYLE = {'background': '#15191d', 'padding': '30px', 'borderRadius': '10px', 'border': '1px solid #2d3436', 'color': 'white', 'marginTop': '20px', 'fontSize': '18px', 'lineHeight': '1.8'}

app.layout = html.Div(style={'backgroundColor': '#0b0e11', 'minHeight': '100vh', 'padding': '40px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[
    # HEADER
    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'borderBottom': '1px solid #333', 'paddingBottom': '20px'}, children=[
        html.H1("NEMESIS T-CORE", style={'color': '#00ffcc', 'margin': '0', 'letterSpacing': '3px'}),
        html.Button("🛑 DETENER SERVICIO", id="btn-stop", n_clicks=0, style={'background': '#ff4757', 'color': 'white', 'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px', 'fontWeight': 'bold', 'cursor': 'pointer'})
    ]),

    # INDICADORES PRINCIPALES
    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginTop': '30px'}, children=[
        html.Div(style=DARK_CARD, children=[html.P("PRECIO BTC/USDT", style={'color': '#888'}), html.H2(id="out-precio", style={'color': '#00d1ff'})]),
        html.Div(style=DARK_CARD, children=[html.P("SEÑAL IA", style={'color': '#888'}), html.H2(id="out-pred")]),
        html.Div(style=DARK_CARD, children=[html.P("INSTRUCCIÓN SALIDA", style={'color': '#888'}), html.H2(id="out-close", style={'color': '#ffa502'})]),
    ]),

    # INDICADORES DE FLUJO
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1fr', 'gap': '20px', 'marginTop': '20px'}, children=[
        html.Div(style=DARK_CARD, children=[html.P("OFI %", style={'color': '#888'}), html.H3(id="out-ofi")]),
        html.Div(style=DARK_CARD, children=[html.P("CVD %", style={'color': '#888'}), html.H3(id="out-cvd-pct")]),
        html.Div(style=DARK_CARD, children=[html.P("ABSORCIÓN", style={'color': '#888'}), html.H3(id="out-manip")]),
        html.Div(style=DARK_CARD, children=[html.P("BALLENAS", style={'color': '#888'}), html.H3(id="out-whales")]),
    ]),

    # --- MANUAL DE OPERACIONES ---
    html.Div(style=MANUAL_STYLE, children=[
        html.H2("📘 MANUAL TÉCNICO DE OPERACIONES", style={'color': '#00ffcc', 'borderBottom': '1px solid #333', 'paddingBottom': '10px'}),
        
        # ESCENARIOS SHORT
        html.Div(style={'marginBottom': '30px'}, children=[
            html.H3("1. Escenarios de VENTA (Buscas un SHORT 🔽)", style={'color': '#ff4757'}),
            html.P("Esta combinación ocurre cuando el precio está en un techo o subiendo por trampa."),
            html.Ul([
                html.Li([html.B("Absorción Crítica + Ballenas (Compra Agresiva):"), html.Br(),
                         "Significado: Existe un esfuerzo masivo de compra a mercado, pero el precio no sube o incluso baja ligeramente.", html.Br(),
                         "Lectura: Los compradores están siendo 'atrapados' por un muro de órdenes limitadas de venta.", html.Br(),
                         "Probabilidad: Alta de caída inmediata."]),
                html.Li([html.B("Absorción + Ballena + CVD (Positivo):"), html.Br(),
                         "Significado: Es un 'Blow-off Top' o techo por agotamiento.", html.Br(),
                         "Lectura: Toda la energía de la sesión ha sido compradora (CVD+), pero en este nivel el precio se ha detenido en seco contra un muro institucional.", html.Br(),
                         "Probabilidad: Muy alta de un giro de tendencia mayor."]),
            ])
        ]),

        # ESCENARIOS LONG
        html.Div(style={'marginBottom': '30px'}, children=[
            html.H3("2. Escenarios de COMPRA (Buscas un LONG 🔼)", style={'color': '#2ecc71'}),
            html.P("Esta combinación ocurre cuando el precio está en un suelo o cayendo por pánico."),
            html.Ul([
                html.Li([html.B("Absorción Crítica + Ballenas (Venta Agresiva):"), html.Br(),
                         "Significado: Existe un esfuerzo masivo de venta a mercado, pero el precio no cae o sube un poco.", html.Br(),
                         "Lectura: Los vendedores están chocando contra un muro de órdenes limitadas de compra (Soporte Institucional).", html.Br(),
                         "Probabilidad: Alta de rebote inminente."]),
                html.Li([html.B("Absorción + Ballena + CVD (Negativo):"), html.Br(),
                         "Significado: Es un pánico vendedor que ha llegado a su fin (Exhaustion).", html.Br(),
                         "Lectura: El mercado ha vendido todo lo que podía durante la sesión (CVD-), pero finalmente ha encontrado una 'cubeta' lo suficientemente grande para absorber toda la oferta.", html.Br(),
                         "Probabilidad: Extrema de un giro alcista violento."]),
            ])
        ]),

        # LÓGICA DE CVD Y TRAMPAS
        html.Div(style={'background': '#1c2227', 'padding': '20px', 'borderRadius': '8px'}, children=[
            html.H3("🧠 Lógica de Contratos y Trampas", style={'color': '#00d1ff'}),
            html.H4("1. El CVD como 'Contratos Atrapados'"),
            html.P("El CVD es el acumulado neto de órdenes a mercado. Si el precio cae pero el CVD es positivo:"),
            html.Ul([
                html.Li("Mucha gente ha estado comprando agresivamente durante la sesión."),
                html.Li("Como el precio no sube, todos esos compradores están en pérdida o 'atrapados'."),
                html.Li("Esos contratos positivos son liquidez que el precio 'irá a buscar': para salir de su posición, esos compradores tendrán que vender (activar sus Stop Loss), lo que acelerará la caída.")
            ]),
            html.H4("2. Análisis del Escenario: 'La Trampa de Toros'"),
            html.P("Cuando NEMESIS muestra Absorción Crítica + Ballenas Comprando y el precio cae:"),
            html.Ul([
                html.Li([html.B("Esfuerzo (CVD +):"), " Hay una intención genuina y masiva de subir el precio."]),
                html.Li([html.B("Resistencia (Absorción):"), " Un vendedor institucional (la 'mano fuerte') está absorbiendo todas esas compras con órdenes limitadas de venta."]),
                html.Li([html.B("Resultado:"), " El precio no sube. Esto confirma que el muro de venta es mucho más grande que el esfuerzo de compra."])
            ]),
            html.H4("3. ¿Por qué le da mayor validez al escenario negativo?"),
            html.P("Le da validez porque crea un desequilibrio de presión:"),
            html.Ul([
                html.Li("Si el CVD fuera negativo, la caída sería 'normal' (gente vendiendo y el precio bajando)."),
                html.Li("Pero con el CVD positivo, la caída es 'explosiva'. Tienes la presión de los vendedores que quieren bajar el precio MÁS la presión de los compradores atrapados que entrarán en pánico y soltarán sus posiciones."),
            ]),
            html.P("En resumen: Esa positividad del CVD representa el 'oxígeno' que los vendedores usarán para quemar el precio hacia abajo.", style={'fontWeight': 'bold', 'color': '#ffa502'})
        ])
    ]),

    # FOOTER / STATUS
    html.Div(id="out-status", style={'textAlign': 'center', 'marginTop': '30px', 'color': '#00ffcc', 'fontSize': '12px', 'opacity': '0.7'}),
    dcc.Interval(id='refresh', interval=500, n_intervals=0)
])

@app.callback(
    [Output("out-precio", "children"), Output("out-pred", "children"), Output("out-pred", "style"), Output("out-close", "children"),
     Output("out-ofi", "children"), Output("out-ofi", "style"), Output("out-cvd-pct", "children"), Output("out-cvd-pct", "style"),
     Output("out-manip", "children"), Output("out-manip", "style"), Output("out-whales", "children"), Output("out-whales", "style"),
     Output("out-status", "children")],
    [Input("refresh", "n_intervals"), Input("btn-stop", "n_clicks")]
)
def update_dashboard(n, n_clicks):
    if n_clicks > 0:
        engine.guardar_memoria_json()
        os._exit(0)
    engine.check_reconnection() 
    c_p = '#2ecc71' if "LONG" in engine.prediction else '#ff4757' if "SHORT" in engine.prediction else '#fff'
    c_o = '#2ecc71' if engine.ofi_pct > 0 else '#ff4757'
    c_c = '#2ecc71' if engine.cvd_pct > 0 else '#ff4757'
    c_m = '#eccc68' if "⚠️" in engine.alert_manipulation else '#fff'
    c_w = '#00ffcc' if "🐋" in engine.alert_whales else '#fff'
    return (f"${engine.price:,.2f}" if engine.price > 0 else "CONECTANDO...", 
            engine.prediction, {'color': c_p, 'fontSize': '32px', 'fontWeight': 'bold'},
            engine.close_instruction, f"{engine.ofi_pct:+.2f}%", {'color': c_o}, 
            f"{engine.cvd_pct:+.2f}%", {'color': c_c}, engine.alert_manipulation, {'color': c_m}, 
            engine.alert_whales, {'color': c_w}, f"STATUS: {engine.status_msg} | TRADES: {engine.trades_count}")

if __name__ == "__main__":
    app.run(debug=False, port=8050, use_reloader=False)