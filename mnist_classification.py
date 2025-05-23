import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Separa as imagens e os rótulos 
x, y = mnist["data"], mnist["target"].astype(np.uint8)
print("Formato dos dados:", x.shape)  # (70000, 784) - 784 é o vetor 28x28 que corresponde a uma imagem de um dígito, com váriações dele
print("Formato dos rótulos:", y.shape)  # (70000)

x_images = x.reshape(-1, 28, 28)

# Imprimindo apenas os x primeiros números que acha pra testar
def mostrar_imagens(imagens, rotulos, qtd=9, salvar_como=None):
    plt.figure(figsize=(6, 6))
    
    for i in range(qtd):
        plt.subplot(3, 3, i + 1)
        plt.imshow(imagens[i], cmap='gray')
        plt.title(f'Dígito: {rotulos[i]}')
        plt.axis('off')
    plt.tight_layout()
    
    if salvar_como:
        plt.savefig(salvar_como)
        print(f"Imagem salva como '{salvar_como}'")
    else:
        plt.show()

mostrar_imagens(x_images, y, salvar_como="amostras_mnist.png")
