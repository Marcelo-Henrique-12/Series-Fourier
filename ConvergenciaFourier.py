import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Definindo a função F(x)
def F(x):
    if -2 <= x < -1:
        return 1
    elif -1 <= x < 1:
        return 0
    elif 1 <= x < 2:
        return 1
    else:
        return 0  # fora do intervalo considerado

# Coeficientes da série de Fourier
def a0(L):
    return (1 / L) * (integrate.quad(lambda x: F(x), -L, L)[0])

def an(n, L):
    return (1 / L) * (integrate.quad(lambda x: F(x) * np.cos(np.pi * n * x / L), -L, L)[0])

def bn(n, L):
    return (1 / L) * (integrate.quad(lambda x: F(x) * np.sin(np.pi * n * x / L), -L, L)[0])

# Soma parcial da série de Fourier
def fourier_series(x, N, L):
    a0_val = a0(L) / 2
    series_sum = a0_val
    for n in range(1, N+1):
        an_val = an(n, L)
        bn_val = bn(n, L)
        series_sum += an_val * np.cos(np.pi*n*x/L) + bn_val * np.sin(np.pi * n * x / L)
    return series_sum

# Parâmetros
T = 4  # Período
L = T / 2  # Metade do período
N = 100  # Número de termos da série de Fourier
x_vals = np.linspace(-2, 2, 1000)

# Valores da função original
F_vals = np.vectorize(F)(x_vals)

# Valor médio da função original
mean_value = np.mean(F_vals)

# Valores da série de Fourier
fourier_vals = fourier_series(x_vals, N, L)

# Pontos de descontinuidade e valores médios dos limites laterais
discontinuities = [-1, 1]
limits = {
    -1: (F(-1-1e-9), F(-1+1e-9)),  # Limites laterais em -1
    1: (F(1-1e-9), F(1+1e-9))      # Limites laterais em 1
}
average_values = {x: (lim[0] + lim[1]) / 2 for x, lim in limits.items()}

# Entrada de pontos específicos para h(x) e plotagem do resultado
while True:
    try:
        input_x = input("Digite um valor de x para avaliar a série de Fourier (ou digite 'sair' para encerrar): ")
        if input_x.lower() == 'sair':
            break
        input_x = float(input_x)
    except ValueError:
        print("Por favor, insira um número válido.")
        continue

    h_x = fourier_series(input_x, N, L)
    print(f"Série de Fourier em x = {input_x}: h(x) = {h_x}")

    # Plotando o ponto especificado no gráfico da série de Fourier e da função original
    plt.figure(figsize=(14, 6))

    # Gráfico da função original
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, F_vals, label='F(x)', color='blue')
    plt.scatter(discontinuities, [F(x) for x in discontinuities], color='red', label='Pontos de Descontinuidade')
    plt.title('Função Original com Descontinuidades')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)

    # Gráfico da série de Fourier
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, fourier_vals, label='Série de Fourier', color='red', linestyle='dashed')
    plt.scatter([input_x], [h_x], color='purple', label=f'h({input_x}) = {h_x}')
    plt.title('Aproximação pela Série de Fourier')
    plt.xlabel('x')
    plt.ylabel('Série de Fourier')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
