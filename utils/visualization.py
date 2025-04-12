import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools

def show_image(img, title=None, cmap=None):
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
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("ruta/a/tu/imagen.jpg"), cv2.COLOR_BGR2RGB)
    show_image(image, title="Original Image")

    heatmap_example = np.random.rand(image.shape[0], image.shape[1])
    overlay_image = overlay_heatmap(image, heatmap_example, alpha=0.5)
    show_image(overlay_image, title="Overlay with Heatmap")

    cm_example = np.array([[50, 10], [5, 35]])
    classes = ["Fresh", "Rotten"]
    plot_confusion_matrix(cm_example, classes, normalize=True)
