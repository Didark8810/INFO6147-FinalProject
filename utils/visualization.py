# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools

def show_image(img, title=None, cmap=None):
    """
    Muestra una imagen utilizando Matplotlib.
    
    Args:
        img (numpy.array): Imagen a mostrar.
        title (str, opcional): Título de la imagen.
        cmap (str, opcional): Colormap a usar (por ejemplo, 'gray').
    """
    plt.figure(figsize=(6, 6))
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Superpone un heatmap sobre una imagen original.
    
    Args:
        image (numpy.array): Imagen original (en formato RGB).
        heatmap (numpy.array): Heatmap normalizado en rango [0, 1].
        alpha (float): Transparencia del heatmap (0 a 1).
        colormap: Código de colormap de OpenCV (por defecto, COLORMAP_JET).
        
    Returns:
        overlay (numpy.array): Imagen resultante con el heatmap superpuesto.
    """
    # Convierte el heatmap a una imagen tipo uint8
    heatmap_uint8 = np.uint8(255 * heatmap)
    # Aplica un colormap al heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    # Convertir de BGR a RGB (OpenCV usa BGR por defecto)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    # Redimensionar heatmap a las dimensiones de la imagen original (si no coinciden)
    heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
    
    # Combinar el heatmap con la imagen original usando la transparencia alpha
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusión',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y dibuja la matriz de confusión.
    La normalización se puede aplicar configurando `normalize=True`.
    
    Args:
        cm (numpy.array): Matriz de confusión.
        classes (list): Lista de nombres de las clases.
        normalize (bool, opcional): Si aplica normalización.
        title (str): Título de la gráfica.
        cmap: Colormap a usar.
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        print("Matriz de confusión normalizada")
    else:
        print('Matriz de confusión, sin normalización')

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Escribir los valores en cada celda
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ejemplo de uso de las funciones:
    
    # 1. Visualizar una imagen (suponiendo que image es un numpy.array en formato RGB)
    import cv2
    # Cargamos una imagen de ejemplo y la convertimos a RGB (por defecto OpenCV carga en BGR)
    image = cv2.cvtColor(cv2.imread("ruta/a/tu/imagen.jpg"), cv2.COLOR_BGR2RGB)
    show_image(image, title="Imagen Original")
    
    # 2. Crear un heatmap falso (solo para ejemplo) y superponerlo a la imagen
    heatmap_example = np.random.rand(image.shape[0], image.shape[1])
    overlay_image = overlay_heatmap(image, heatmap_example, alpha=0.5)
    show_image(overlay_image, title="Overlay con Heatmap")
    
    # 3. Visualizar una matriz de confusión de ejemplo
    cm_example = np.array([[50, 10], [5, 35]])
    classes = ["Fresh", "Rotten"]
    plot_confusion_matrix(cm_example, classes, normalize=True)
