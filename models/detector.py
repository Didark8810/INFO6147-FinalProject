# detector.py

import torch
from torchvision import models, transforms
from PIL import Image


class FruitDetector:
    def __init__(self, score_threshold=0.8, device=None):
        """
        Inicializa el detector de frutas usando Faster R-CNN preentrenado.

        Args:
            score_threshold (float): Puntuación mínima para considerar una detección válida.
            device (str, optional): Dispositivo para correr el modelo ("cuda" o "cpu"). 
                                    Si es None, se usará cuda si está disponible.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Cargamos el modelo preentrenado de Faster R-CNN, entrenado en el dataset COCO.
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
        # Transformación básica: Convertir una imagen PIL a tensor.
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.score_threshold = score_threshold

    def detect(self, pil_image):
        """
        Detecta regiones de interés en una imagen.

        Args:
            pil_image (PIL.Image): Imagen de entrada en formato PIL.

        Returns:
            boxes (list): Lista de bounding boxes filtrados (cada caja es [xmin, ymin, xmax, ymax]).
        """
        # Convertir la imagen PIL a un tensor
        image_tensor = self.transform(pil_image).to(self.device)
        
        # Se ejecuta el modelo (recordar que espera una lista de tensores)
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        boxes = predictions['boxes']
        scores = predictions['scores']
        
        # Filtramos las detecciones que tengan un score mayor al umbral especificado
        valid_idxs = scores > self.score_threshold
        filtered_boxes = boxes[valid_idxs]
        
        # Regresar las cajas como un array de NumPy para facilitar su manejo
        return filtered_boxes.cpu().numpy()


# Ejemplo de uso al ejecutar este módulo directamente
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import sys

    # Se espera que se provea una ruta a la imagen de prueba
    if len(sys.argv) < 2:
        print("Uso: python detector.py <ruta_de_la_imagen>")
        sys.exit(1)

    image_path = sys.argv[1]
    pil_img = Image.open(image_path).convert("RGB")
    
    detector = FruitDetector(score_threshold=0.8)
    boxes = detector.detect(pil_img)
    
    # Visualización de la imagen y las cajas detectadas
    fig, ax = plt.subplots(1)
    ax.imshow(pil_img)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
