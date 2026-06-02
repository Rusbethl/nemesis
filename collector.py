import asyncio
import json
import os
import sys
import signal
import pandas as pd
from websockets import connect
from datetime import datetime
import config

# --- Parámetros de Configuración Global ---
SYMBOL = "btcusdt"
# URL del WebSocket de Binance: captura trades agregados y actualizaciones de profundidad (Order Book)
URL_WSS = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade/{SYMBOL}@depth@100ms"
# Archivo de persistencia: garantiza que el contador de eventos no se resetee al cerrar el programa
STATE_FILE = "nemesis_metadata.json" 

class NemesisCollector:
    def __init__(self):
        """
        Constructor de la clase. Inicializa estructuras de datos, carga el estado
        previo y configura la captura de señales del sistema para cierres seguros.
        """
        self.buffer = []          # Almacenamiento temporal en memoria para reducir escritura en disco
        self.running = True       # Control de flujo del bucle principal
        
        # Recuperación de estado: permite la continuidad del proyecto (Meta de ahorro de datos)
        self.sesion_total = self.cargar_estado_previo()
        
        # Gestión de señales (SIGINT = Ctrl+C, SIGTERM = Cierre de sistema)
        # Esto evita que el programa se interrumpa abruptamente y corrompa archivos
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def cargar_estado_previo(self):
        """
        Lee el archivo JSON de metadatos. Si existe, recupera el conteo total
        de eventos recolectados en sesiones pasadas para dar continuidad histórica.
        """
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f).get("sesion_total", 0)
            except Exception: 
                return 0 # En caso de archivo corrupto, inicia desde cero por seguridad
        return 0

    def guardar_estado(self):
        """
        Persistencia de metadatos: Registra el progreso actual en el disco duro.
        Esto permite que el Processor y el Collector estén sincronizados en volumen.
        """
        with open(STATE_FILE, 'w') as f:
            json.dump({"sesion_total": self.sesion_total, "last_run": str(datetime.now())}, f)

    def handle_exit(self, signum, frame):
        """
        Intercepta la orden de cierre del usuario para ejecutar una rutina
        de limpieza (guardado de buffer residual) antes de finalizar.
        """
        print(f"\n[!] Interrupción detectada. Asegurando datos finales...")
        self.detener()

    def guardar_batch(self):
        """
        Lógica de Escritura Atómica (.tmp -> .parquet):
        1. Crea un archivo temporal (.tmp) para evitar que el Processor lea datos incompletos.
        2. Al finalizar la escritura, lo renombra a .parquet (operación instantánea a nivel SO).
        """
        if not self.buffer: return
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_final = os.path.join(config.DIR_RAW, f"batch_{timestamp_str}.parquet")
            path_temp = os.path.join(config.DIR_RAW, f"batch_{timestamp_str}.tmp")
            
            # Conversión de lista de diccionarios a DataFrame de Pandas
            df = pd.DataFrame(self.buffer)
            # Compresión Snappy: equilibrio óptimo entre velocidad y espacio en disco
            df.to_parquet(path_temp, engine='pyarrow', compression='snappy')
            
            # Sustitución atómica: el archivo solo aparece para el Processor cuando está 100% escrito
            os.replace(path_temp, path_final)
            
            self.guardar_estado() # Actualiza el contador histórico en cada guardado exitoso
            self.buffer = []      # Libera memoria RAM
            return True
        except Exception as e:
            # Limpieza preventiva: si falla el guardado, elimina el archivo temporal
            if 'path_temp' in locals() and os.path.exists(path_temp): os.remove(path_temp)
            print(f"Error al guardar: {e}")
            return False

    async def iniciar(self):
        """
        Punto de entrada asíncrono. Gestiona la conexión WebSocket y el procesamiento
        de mensajes en tiempo real sincronizados con el reloj del Exchange.
        """
        print(f"🚀 NÉMESIS COLLECTOR | Continuidad: {self.sesion_total:,} eventos previos.")
        
        while self.running:
            try:
                async with connect(URL_WSS) as websocket:
                    print("📶 Conexión establecida con Binance Futures.")
                    while self.running:
                        msg = await websocket.recv()
                        data = json.loads(msg)
                        
                        # Sincronización Temporal de Precisión (Exchange Time 'E')
                        # Se utiliza el timestamp de Binance para evitar el ruido de la latencia local
                        event_time = pd.to_datetime(data.get("E"), unit='ms') if data.get("E") else datetime.now()

                        # Identificación del lado de la transacción (Lógica de Microestructura)
                        side = None
                        if data.get("e") == "aggTrade":
                            side = "sell" if data.get("m") else "buy"

                        # Formación del evento: Estructura compatible con el etiquetador y modelos de IA
                        evento = {
                            "timestamp": event_time,
                            "type": data.get("e"),
                            "side": side,
                            "price": data.get("p") if data.get("e") == "aggTrade" else None,
                            "quantity": data.get("q") if data.get("e") == "aggTrade" else None,
                            "bids": str(data.get("b")) if data.get("e") == "depthUpdate" else None,
                            "asks": str(data.get("a")) if data.get("e") == "depthUpdate" else None
                        }

                        self.buffer.append(evento)
                        self.sesion_total += 1
                        
                        # Monitorización de sesión en tiempo real
                        sys.stdout.write(f"\r📥 Buffer: {len(self.buffer)} | Total Histórico: {self.sesion_total:,}")
                        sys.stdout.flush()

                        # Umbral de guardado: garantiza que no se pierdan más de 1000 eventos ante fallos
                        if len(self.buffer) >= 1000: self.guardar_batch()
            except Exception:
                # Reconexión automática en caso de pérdida de internet
                await asyncio.sleep(5)

    def detener(self):
        """Finaliza el ciclo de recolección y asegura la persistencia final."""
        self.running = False
        if self.buffer: self.guardar_batch()
        self.guardar_estado()
        sys.exit(0)

if __name__ == "__main__":
    collector = NemesisCollector()
    # Ejecución del bucle de eventos asíncronos
    asyncio.get_event_loop().run_until_complete(collector.iniciar())