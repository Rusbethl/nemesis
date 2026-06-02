import torch
import torch.nn as nn
import torch.optim as optim

class NemesisCNN(nn.Module):
    """
    Arquitectura de Red Neuronal Profunda (MLP).
    Este modelo es el estándar en la industria financiera para procesar 
    señales de microestructura con baja latencia.
    """
    def __init__(self, input_dim):
        super(NemesisCNN, self).__init__()
        # Definición de capas: Cada capa aprende patrones más complejos
        self.network = nn.Sequential(
            # Capa de entrada: Procesa los indicadores (OFI, CVD, etc.)
            nn.Linear(input_dim, 64),
            nn.ReLU(), # Función de activación para detectar relaciones no lineales
            
            # Capa intermedia: Abstracción de la presión del mercado
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Capa de salida: Genera la probabilidad de éxito (0 a 1)
            nn.Linear(32, 1),
            nn.Sigmoid() # Escala la salida a un rango probabilístico
        )

    def forward(self, x):
        """Paso de inferencia: Transforma señales en predicciones."""
        return self.network(x)

    def fit(self, X, y, lr=0.001):
        """
        Función de entrenamiento (Backpropagation).
        Permite el ajuste fino de las neuronas basándose en el error cometido.
        """
        # Detección automática de hardware (Acelera el proceso en Big Data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Conversión de datos a tensores (Formato nativo de PyTorch)
        X_tensor = torch.tensor(X.values.astype('float32')).to(device)
        y_tensor = torch.tensor(y.values.astype('float32')).view(-1, 1).to(device)
        
        # Función de pérdida (Binary Cross Entropy): Mide la distancia entre predicción y realidad
        criterion = nn.BCELoss()
        # Optimizador Adam: Ajusta el conocimiento de forma adaptativa
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        self.train() 
        optimizer.zero_grad() # Reinicia gradientes para el nuevo lote
        
        # Proceso de aprendizaje
        outputs = self(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward() # Calcula el impacto del error en cada neurona
        optimizer.step() # Actualiza el cerebro con el nuevo conocimiento
        
        return loss.item()