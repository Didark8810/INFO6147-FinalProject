import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math
from utils.coco_classes import CLASSES
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import os


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h
    ], dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    return b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

def detect(im, model, transform):
    img = transform(im).unsqueeze(0)
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]
    print(f"Boxes shape: {boxes}")
    banana_idx = 55
    keep = (probas.argmax(-1) == banana_idx) & (probas.max(-1).values > 0.9)
    print(f"Keep: {keep}")
    print(f"IMAGE: {im.size}")
    bboxes_scaled = rescale_bboxes(boxes[keep], im.size)
    print(f"Boxes ESCALADO: {bboxes_scaled}")
    return probas[keep], bboxes_scaled

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    width, height = pil_img.size
    center_x, center_y = width // 2, height // 2

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        box_center_x = (xmin + xmax) / 2
        box_center_y = (ymin + ymax) / 2
        ax.plot([center_x, box_center_x], [center_y, box_center_y], color='green', linewidth=2)
        distance = math.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2)
        mid_x = (center_x + box_center_x) / 2
        mid_y = (center_y + box_center_y) / 2
        ax.text(mid_x, mid_y, f'{distance:.1f}px', color='lime', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    ax.plot(center_x, center_y, marker='o', color='white', markersize=10)
    plt.axis('off')
    plt.show()


def plot_results_to_image(pil_img, prob, boxes, save_path="outputs/detection_result.jpg"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(pil_img)
    width, height = pil_img.size
    center_x, center_y = width // 2, height // 2

    print(boxes)

    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        box_center_x = (xmin + xmax) / 2
        box_center_y = (ymin + ymax) / 2
        ax.plot([center_x, box_center_x], [center_y, box_center_y], color='green', linewidth=2)
        distance = math.sqrt((center_x - box_center_x) ** 2 + (center_y - box_center_y) ** 2)
        mid_x = (center_x + box_center_x) / 2
        mid_y = (center_y + box_center_y) / 2
        ax.text(mid_x, mid_y, f'{distance:.1f}px', color='lime', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    ax.plot(center_x, center_y, marker='o', color='white', markersize=10)
    ax.axis('off')

    # Guardar imagen localmente
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Asegura que la carpeta exista
    plt.savefig(save_path, format="jpeg", bbox_inches="tight")  # ← ESTA LÍNEA GUARDA EN DISCO

    # Convertir a imagen en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
