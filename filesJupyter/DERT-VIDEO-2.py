#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import requests
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}



detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # Preprocesamiento
    img = transform(im).unsqueeze(0)

    # Asegura que la imagen no excede 1600 píxeles por lado
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # Pasa la imagen por el modelo
    outputs = model(img)

    # Probabilidades de clase
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]

    # Índice de clase de banana
    banana_idx = 55

    # Filtra por clase banana y confianza mayor a 0.9
    keep = (probas.argmax(-1) == banana_idx) & (probas.max(-1).values > 0.9)

    # Ajusta las cajas al tamaño original de la imagen
    bboxes_scaled = rescale_bboxes(boxes[keep], im.size)
    return probas[keep], bboxes_scaled


#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#im = Image.open(requests.get(url, stream=True).raw)
im = Image.open("uno/f7.jpg")

scores, boxes = detect(im, detr, transform)


import math  # para calcular distancia

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    # Centro de la imagen
    width, height = pil_img.size
    center_x = width // 2
    center_y = height // 2

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        # Dibuja la caja
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        
        # Texto de clase
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

        # Centro de la caja
        box_center_x = (xmin + xmax) / 2
        box_center_y = (ymin + ymax) / 2

        # Dibuja línea verde desde el centro de la imagen al centro de la caja
        ax.plot([center_x, box_center_x], [center_y, box_center_y], color='green', linewidth=2)

        # Calcula distancia euclidiana
        distance = math.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2)

        # Escribe la distancia en la mitad de la línea
        mid_x = (center_x + box_center_x) / 2
        mid_y = (center_y + box_center_y) / 2
        ax.text(mid_x, mid_y, f'{distance:.1f}px', color='lime', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.5))

    # Punto blanco en el centro de la imagen
    ax.plot(center_x, center_y, marker='o', color='white', markersize=10)

    plt.axis('off')
    plt.show()


plot_results(im, scores, boxes)


# In[2]:


import cv2
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# Asegúrate de que tu clase DETRdemo y el modelo ya estén definidos
# Y que esté este bloque ejecutado:
# detr = DETRdemo(num_classes=91)
# detr.load_state_dict(...)
# detr.eval()

# Transformación de imagen
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Reutilizamos estas funciones del código 2
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# Detección en un solo frame
def detect(im, model, transform):
    img = transform(im).unsqueeze(0)
    outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]

    # Índice de clase "orange"
    orange_idx = CLASSES.index('orange')

    # Filtro: confianza > 0.7 Y clase más probable = "orange"
    scores = probas[:, orange_idx]
    keep = (scores > 0.7) & (probas.argmax(-1) == orange_idx)

    bboxes_scaled = rescale_bboxes(boxes[keep], im.size)
    return probas[keep], bboxes_scaled


# Visualizar resultados sobre frame
def draw_boxes(frame, prob, boxes):
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        label = CLASSES[cl]
        conf = p[cl].item()
        if conf > 0.7:
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            text = f'{label}: {conf:.2f}'
            cv2.putText(frame, text, (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

# Cargar video
cap = cv2.VideoCapture('video1.mp4')  # Cambia por la ruta de tu video

import signal
import sys

# Permitir interrupción con Ctrl+C
def signal_handler(sig, frame):
    print('Cerrando...')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Obtener FPS del video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # Cada cuántos frames ejecutar la detección
print(frame_interval)

frame_count = 0
last_scores, last_boxes = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    center_x, center_y = W // 2, H // 2

    # Convertir a RGB y PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Solo ejecutar detección cada 1 segundo
    if frame_count % frame_interval == 0:
        last_scores, last_boxes = detect(pil_img, detr, transform)

    scores, boxes = last_scores, last_boxes

    # Dibujar cajas
    frame_out = draw_boxes(frame, scores, boxes)

    # Punto rojo en el centro
    cv2.circle(frame_out, (center_x, center_y), 5, (0, 0, 255), -1)

    # Dibujar líneas y distancias
    if boxes is not None:
        for box in boxes.tolist():
            xmin, ymin, xmax, ymax = box
            box_center_x = int((xmin + xmax) / 2)
            box_center_y = int((ymin + ymax) / 2)

            cv2.line(frame_out, (center_x, center_y), (box_center_x, box_center_y), (0, 255, 0), 2)

            dx = box_center_x - center_x
            dy = box_center_y - center_y
            distance = np.sqrt(dx**2 + dy**2)

            dist_text = f'{int(distance)} px'
            mid_x = (center_x + box_center_x) // 2
            mid_y = (center_y + box_center_y) // 2
            cv2.putText(frame_out, dist_text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Mostrar
    cv2.imshow('DETR Video', frame_out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('DETR Video', cv2.WND_PROP_VISIBLE) < 1:
        print("Ventana cerrada o 'q' presionado.")
        break

    frame_count += 1

# Liberar recursos y cerrar ventana específica al finalizar
cap.release()

# Cerrar la ventana que muestra el video
cv2.destroyWindow('DETR Video')  # Esta línea es opcional pero buena práctica

# Cerrar cualquier otra ventana abierta por OpenCV, por si acaso
cv2.destroyAllWindows()




# In[3]:





# In[ ]:




