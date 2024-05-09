import torch
import torch.nn as nn

class FCN(nn.Module):  
    """
    Red Neuronal Totalmente Conectada (Fully Connected Network, FCN).

    Parámetros:
    N_INPUT (int): Número de entradas de la red.
    N_OUTPUT (int): Número de salidas de la red.
    N_HIDDEN (int): Número de unidades ocultas en cada capa.
    N_LAYERS (int): Número de capas ocultas en la red.

    Atributos:
    fcs (torch.nn.Sequential): Capa de entrada de la red.
    fch (torch.nn.Sequential): Capas ocultas de la red.
    fce (torch.nn.Linear): Capa de salida de la red.
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        """
        Inicializa la red con los parámetros dados.
        """
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
                    nn.Sequential(*[
                        nn.Linear(N_HIDDEN, N_HIDDEN),
                        activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        """
        Define cómo se debe calcular la salida de la red a partir de la entrada x.

        Parámetros:
        x (torch.Tensor): Tensor de entrada a la red.

        Devuelve:
        output (torch.Tensor): Tensor de salida de la red.
        """
        a = x * (x - torch.pi)
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)        
        output = torch.einsum('ij,ij->ij', a, x)
        return output
    

import torch.nn.functional as F
class CNNPDE(nn.Module):
    """
    Red Neuronal Convolucional (Convolutional Neural Network, CNN) .    

    Atributos:
    conv1 (torch.nn.Conv1d): Primera capa convolucional de la red.
    conv2 (torch.nn.Conv1d): Segunda capa convolucional de la red.
    fc (torch.nn.Linear): Capa totalmente conectada de la red.
    """

    def __init__(self):
        """
        Inicializa la red con las capas convolucionales y totalmente conectada.
        """
        super(CNNPDE, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        """
        Define cómo se debe calcular la salida de la red a partir de la entrada x.

        Parámetros:
        x (torch.Tensor): Tensor de entrada a la red.

        Devuelve:
        output (torch.Tensor): Tensor de salida de la red.
        """        
        a = x * (x - torch.pi)
        x = x.unsqueeze(1)  
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = F.avg_pool1d(x, x.size()[2])        
        x = x.permute(0, 2, 1)  
        output = self.fc(x)
        output = torch.einsum('ij,ij->ij', a, output.squeeze(1))
        return output
    
class RNNPDE(nn.Module):
    """
    Red Neuronal Recurrente (Recurrent Neural Network, RNN).     

    Atributos:
    rnn (torch.nn.RNN): Capa recurrente de la red.
    fc (torch.nn.Sequential): Capa totalmente conectada de la red.

    """

    def __init__(self):
        """
        Inicializa la red con la capa recurrente y la capa totalmente conectada.
        """
        super(RNNPDE, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Sequential(
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """
        Define cómo se debe calcular la salida de la red a partir de la entrada x.

        Parámetros:
        x (torch.Tensor): Tensor de entrada a la red.

        Devuelve:
        output (torch.Tensor): Tensor de salida de la red.
        """
        a = x * (x - torch.pi)
        x = x.unsqueeze(2)  
        out, _ = self.rnn(x) 
        out = out[:, -1, :]  
        out = self.fc(out) 
        out = torch.einsum('ij,ij->ij', a, out)
        return out
    
class TransformerEDP(nn.Module):    
    """
    Clase TransformerEDP que hereda de nn.Module. Esta clase implementa una arquitectura de red neuronal Transformer para la resolución de ecuaciones diferenciales parciales (EDP).

    Parámetros:
    input_dim (int): Dimensión de la entrada de la red.
    output_dim (int): Dimensión de la salida de la red.
    num_layers (int): Número de capas del codificador Transformer.
    hidden_dim (int): Dimensión de las capas ocultas de la red.

    Atributos:
    encoder (torch.nn.Sequential): Capa de codificación de la red, compuesta por una capa lineal y una activación Tanh.
    transformer_encoder (torch.nn.TransformerEncoder): Codificador Transformer, compuesto por varias capas TransformerEncoderLayer.
    decoder (torch.nn.Sequential): Capa de decodificación de la red, compuesta por una capa lineal.
    """

    def __init__(self, input_dim, output_dim, num_layers, hidden_dim):
        """
        Inicializa la red con los parámetros dados.
        """
        super(TransformerEDP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, activation='relu'), 
            num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):  
        """
        Define cómo se debe calcular la salida de la red a partir de la entrada x.

        Parámetros:
        x (torch.Tensor): Tensor de entrada a la red.

        Devuelve:
        output (torch.Tensor): Tensor de salida de la red.
        """
        a = x * (x - torch.pi)      
        x = self.encoder(x)  
        output = self.transformer_encoder(x)  
        output = self.decoder(output)
        out = torch.einsum('ij,ij->ij', a, output)
        return out  