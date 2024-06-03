import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Função periódica e descontínua
def F(x):
    # Ajustar x para o intervalo [-2, 2]
    #
    x = (x + 2) % 4 - 2
    
    if -2 <= x < -1:
        return 1
    elif -1 <= x < 1:
        return 0
    elif 1 <= x < 2:
        return 1
    else:
        return 0 
    
    
# Coeficientes da série de Fourier
def a0(L):
    return (1 / L) * (integrate.quad(lambda x: F(x), -L, L)[0])

def an(n, L):
    return (1 / L) * (integrate.quad(lambda x: F(x) * np.cos(np.pi * n * x / L), -L, L)[0])

def bn(n, L):
    return (1 / L) * (integrate.quad(lambda x: F(x) * np.sin(np.pi * n * x / L), -L, L)[0])

# Soma parcial da série de Fourier
def serie_fourier(x, N, L):
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
x_vals = np.linspace(-4, 4, 1000)

# Valores da função original
F_vals = np.vectorize(F)(x_vals)

# Valores da série de Fourier
fourier_vals = serie_fourier(x_vals, N, L)


# Entrada de pontos específicos para h(x) e plotagem do resultado
while True:
    try:
        input_x = input("Digite um valor de x para avaliar a série de Fourier (ou digite 'sair' para encerrar): ")
        if input_x.lower() == 'sair':
            break
        input_x = float(input_x)
        if input_x < -4 or input_x > 4:
            print("Por favor, insira um valor de x dentro do intervalo [-4, 4].")
            continue
    except ValueError:
        print("Por favor, insira um número válido.")
        continue
    
    #Calcula a série de Fourier para o ponto X enviado
    h_x = serie_fourier(input_x, N, L)
    print(f"Série de Fourier em x = {input_x}: h(x) = {h_x}")

    # Plotando o ponto especificado no gráfico da série de Fourier e da função original
    plt.figure(figsize=(14, 6))

    # Gráfico da função original
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, F_vals, label='F(x)', color='blue')
    
    # Enfatiza os pontos de descontinuidade
    for i in range(1, len(F_vals)):
        if (F_vals[i - 1] == 0 and F_vals[i] == 1) or (F_vals[i - 1] == 1 and F_vals[i] == 0):
            plt.scatter(x_vals[i - 1], 0, color='red', marker='o', s=100) 
            plt.scatter(x_vals[i], 1, color='red', edgecolors='red', marker='o', s=100)     
            
    plt.title('Função Original com Descontinuidades')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend()
    plt.grid(True)

    # Gráfico da série de Fourier
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, fourier_vals, label='Série de Fourier', color='red', linestyle='dashed')
    plt.scatter([input_x], [h_x], color='purple', label=f'h({input_x}) = {h_x}') # coloca o Ponto aproximado para o valor de x inserido
    plt.title('Aproximação pela Série de Fourier')
    plt.xlabel('x')
    plt.ylabel('Série de Fourier')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
