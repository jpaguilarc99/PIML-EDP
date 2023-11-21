import numpy as np
import matplotlib.pyplot as plt

# Función para discretizar la ecuación de onda 1D utilizando diferencias finitas
def discretize_wave_equation(u, u_prev, dt, dx, c):
    # Calcula el número de puntos en la malla espacial
    N_x = len(u)
    
    # Copia la función actual en un nuevo arreglo para almacenar la solución en el siguiente paso de tiempo
    u_next = np.copy(u)
    
    # Itera sobre todos los puntos espaciales, excepto los bordes, para aplicar las diferencias finitas
    for i in range(1, N_x - 1):
        # Aplica la diferencia finita en la derivada espacial y temporal
        u_next[i] = 2*u[i] - u_prev[i] + (c*dt/dx)**2 * (u[i+1] - 2*u[i] + u[i-1])
    
    return u_next

# Función para calcular la función de pérdida (MSE) entre la solución numérica y una solución objetivo
def loss_function(u_numeric, u_target):
    return np.mean((u_numeric - u_target)**2)

# Parámetros de la simulación
L = 1.0         # Longitud de la cuerda
N_x = 100       # Número de puntos espaciales
T = 1.0         # Tiempo total de simulación
N_t = 80000      # Número de pasos de tiempo
c = 1.0         # Velocidad de propagación de la onda

# Intervalos espacial y temporal
dx = L / (N_x - 1)
dt = T / N_t

# Condiciones iniciales: una función de onda triangular
x = np.linspace(0, L, N_x)
u_initial = np.where(x < L/2, x/(L/2), (L-x)/(L/2))

# Solución objetivo (para demostración): una función de onda triangular desplazada en el tiempo
u_target = np.where((x - c*T/2) < L/2, (x - c*T/2)/(L/2), (L-(x - c*T/2))/(L/2))

# Iteración temporal utilizando el método de diferencias finitas
u_current = np.copy(u_initial)
u_previous = np.copy(u_initial)
for t in range(N_t):
    u_next = discretize_wave_equation(u_current, u_previous, dt, dx, c)
    u_previous = u_current
    u_current = u_next

# Cálculo de la función de pérdida
loss = loss_function(u_current, u_target)

print("Valor de la función de pérdida (MSE):", loss)

# Graficar la solución real y la solución numérica
plt.figure(figsize=(10, 6))
plt.plot(x, u_target, label='Solución real', linestyle='dashed', color='blue')
plt.plot(x, u_current, label='Solución numérica', color='red')
plt.xlabel('Posición (x)')
plt.ylabel('Amplitud (u)')
plt.title('Comparación de la solución real y la solución numérica')
plt.legend()
plt.grid(True)
plt.show()